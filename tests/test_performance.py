"""Tests for analytics/performance.py.

Hand-calculated where possible, with return series chosen so the arithmetic
is exact and independently checkable (not just "recompute the
implementation's own formula with different code").
"""

import math
from datetime import datetime, timedelta

import pytest

from engine.events import FillEvent
from engine.portfolio import PortfolioSnapshot
from analytics.performance import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    DrawdownResult,
    PerformanceError,
    RoundTrip,
    average_win_loss,
    cagr,
    extract_round_trips,
    format_tearsheet,
    generate_tearsheet,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

T0 = datetime(2024, 1, 2, 9, 30)


def ts(n: int) -> datetime:
    return T0 + timedelta(days=n)


def snap(t: datetime, total_value: float, positions_value: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(timestamp=t, cash=total_value - positions_value, positions_value=positions_value)


def fill(t: datetime, direction: str, quantity: int, price: float, commission: float = 1.0, symbol: str = "AAPL") -> FillEvent:
    return FillEvent(timestamp=t, symbol=symbol, quantity=quantity, direction=direction, fill_price=price, commission=commission)


class TestTotalReturn:
    def test_positive_return(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1100.0)]
        assert total_return(curve) == pytest.approx(0.10)

    def test_negative_return(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 750.0)]
        assert total_return(curve) == pytest.approx(-0.25)

    def test_raises_on_single_point(self):
        with pytest.raises(PerformanceError):
            total_return([snap(ts(0), 1000.0)])

    def test_raises_on_empty_curve(self):
        with pytest.raises(PerformanceError):
            total_return([])

    def test_raises_on_zero_starting_value(self):
        with pytest.raises(PerformanceError):
            total_return([snap(ts(0), 0.0), snap(ts(1), 100.0)])


class TestCAGR:
    def test_one_year_doubling_is_one_hundred_percent(self):
        curve = [snap(T0, 1000.0), snap(T0 + timedelta(days=CALENDAR_DAYS_PER_YEAR), 2000.0)]
        assert cagr(curve) == pytest.approx(1.0)

    def test_two_year_quadrupling_is_one_hundred_percent(self):
        curve = [snap(T0, 1000.0), snap(T0 + timedelta(days=2 * CALENDAR_DAYS_PER_YEAR), 4000.0)]
        assert cagr(curve) == pytest.approx(1.0)

    def test_one_year_halving_is_negative_fifty_percent(self):
        curve = [snap(T0, 1000.0), snap(T0 + timedelta(days=CALENDAR_DAYS_PER_YEAR), 500.0)]
        assert cagr(curve) == pytest.approx(-0.5)

    def test_raises_on_single_point(self):
        with pytest.raises(PerformanceError):
            cagr([snap(ts(0), 1000.0)])


class TestSharpeRatio:
    def test_zero_mean_excess_return_is_zero_via_genuine_calculation(self):
        # returns [+0.01, -0.01]: nonzero stdev, but mean excess is exactly
        # 0, so this exercises the real division (0/x), not the
        # zero-deviation fallback branch.
        curve = [snap(ts(0), 100.0), snap(ts(1), 101.0), snap(ts(2), 99.99)]
        assert sharpe_ratio(curve) == pytest.approx(0.0)

    def test_flat_curve_hits_the_zero_volatility_fallback(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1000.0), snap(ts(2), 1000.0)]
        assert sharpe_ratio(curve) == 0.0

    def test_known_nonzero_sharpe(self):
        # returns [0.01, 0.03]: deviations of +-0.01 from the mean (0.02)
        # make daily Sharpe exactly sqrt(2), so the annualized value is
        # exactly sqrt(2 * TRADING_DAYS_PER_YEAR).
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1010.0), snap(ts(2), 1040.3)]
        expected = math.sqrt(2 * TRADING_DAYS_PER_YEAR)
        assert sharpe_ratio(curve) == pytest.approx(expected)

    def test_higher_risk_free_rate_strictly_lowers_sharpe(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1010.0), snap(ts(2), 1040.3)]
        assert sharpe_ratio(curve, risk_free_rate=0.05) < sharpe_ratio(curve, risk_free_rate=0.0)

    def test_raises_on_single_point(self):
        with pytest.raises(PerformanceError):
            sharpe_ratio([snap(ts(0), 1000.0)])


class TestSortinoRatio:
    def test_known_nonzero_sortino(self):
        # returns [0.02, 0.02, -0.02, 0.02]: downside deviation over all 4
        # periods works out to exactly 0.01 (== the mean return), so the
        # daily ratio collapses to exactly 1.0 and the annualized value is
        # exactly sqrt(TRADING_DAYS_PER_YEAR).
        curve = [
            snap(ts(0), 1000.0), snap(ts(1), 1020.0), snap(ts(2), 1040.4),
            snap(ts(3), 1019.592), snap(ts(4), 1039.98384),
        ]
        assert sortino_ratio(curve) == pytest.approx(math.sqrt(TRADING_DAYS_PER_YEAR))

    def test_all_positive_returns_hits_the_zero_downside_fallback(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1010.0), snap(ts(2), 1030.2), snap(ts(3), 1061.106)]
        assert sortino_ratio(curve) == 0.0

    def test_raises_on_single_point(self):
        with pytest.raises(PerformanceError):
            sortino_ratio([snap(ts(0), 1000.0)])


class TestMaxDrawdown:
    """values [100, 120, 90, 95, 130, 80, 110]:

    peak climbs to 120 (t1), then to 130 (t4); the worst drawdown is
    80/130 - 1 = -5/13 (~-38.46%) at t5, off the t4 peak. The curve never
    closes back above 130 by t6, so it's unrecovered.
    """

    def test_peak_trough_and_no_recovery(self):
        values = [100, 120, 90, 95, 130, 80, 110]
        curve = [snap(ts(n), v) for n, v in enumerate(values)]

        result = max_drawdown(curve)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct == pytest.approx(-5 / 13)
        assert result.peak_timestamp == ts(4)
        assert result.trough_timestamp == ts(5)
        assert result.duration_days == 1
        assert result.recovered is False
        assert result.recovery_timestamp is None

    def test_recovery_is_detected_when_curve_closes_above_prior_peak(self):
        values = [100, 120, 90, 95, 130, 80, 110, 135]
        curve = [snap(ts(n), v) for n, v in enumerate(values)]

        result = max_drawdown(curve)

        assert result.recovered is True
        assert result.recovery_timestamp == ts(7)

    def test_monotonically_rising_curve_has_zero_drawdown(self):
        curve = [snap(ts(n), 100.0 + n) for n in range(5)]
        result = max_drawdown(curve)
        assert result.max_drawdown_pct == 0.0

    def test_raises_on_single_point(self):
        with pytest.raises(PerformanceError):
            max_drawdown([snap(ts(0), 1000.0)])


class TestExtractRoundTrips:
    def test_simple_win_and_loss(self):
        trades = [
            fill(ts(0), "BUY", 10, 100.0, commission=1.0),
            fill(ts(1), "SELL", 10, 110.0, commission=1.0),
            fill(ts(2), "BUY", 5, 50.0, commission=1.0),
            fill(ts(3), "SELL", 5, 40.0, commission=1.0),
        ]

        round_trips = extract_round_trips(trades)

        assert len(round_trips) == 2
        win, loss = round_trips
        assert isinstance(win, RoundTrip)
        assert win.pnl == pytest.approx(98.0)  # (110-100)*10 - 1 - 1
        assert loss.pnl == pytest.approx(-52.0)  # (40-50)*5 - 1 - 1

    def test_fifo_matches_across_multiple_open_lots(self):
        trades = [
            fill(ts(0), "BUY", 10, 100.0, commission=0.0),
            fill(ts(1), "BUY", 10, 110.0, commission=0.0),
            fill(ts(2), "SELL", 15, 120.0, commission=0.0),  # closes lot 1 fully, lot 2 partially
            fill(ts(3), "SELL", 5, 130.0, commission=0.0),   # closes the remainder of lot 2
        ]

        round_trips = extract_round_trips(trades)

        assert len(round_trips) == 3
        rt1, rt2, rt3 = round_trips
        assert (rt1.quantity, rt1.entry_price, rt1.pnl) == (10, 100.0, pytest.approx(200.0))
        assert (rt2.quantity, rt2.entry_price, rt2.pnl) == (5, 110.0, pytest.approx(50.0))
        assert (rt3.quantity, rt3.entry_price, rt3.pnl) == (5, 110.0, pytest.approx(100.0))

    def test_sell_exceeding_open_lots_raises(self):
        trades = [
            fill(ts(0), "BUY", 5, 100.0),
            fill(ts(1), "SELL", 10, 110.0),
        ]
        with pytest.raises(PerformanceError):
            extract_round_trips(trades)

    def test_empty_trade_log_yields_no_round_trips(self):
        assert extract_round_trips([]) == []


class TestWinRateAndAverageWinLoss:
    def test_mixed_win_loss(self):
        trades = [
            fill(ts(0), "BUY", 10, 100.0, commission=1.0),
            fill(ts(1), "SELL", 10, 110.0, commission=1.0),
            fill(ts(2), "BUY", 5, 50.0, commission=1.0),
            fill(ts(3), "SELL", 5, 40.0, commission=1.0),
        ]
        round_trips = extract_round_trips(trades)

        assert win_rate(round_trips) == pytest.approx(0.5)
        avg_win, avg_loss = average_win_loss(round_trips)
        assert avg_win == pytest.approx(98.0)
        assert avg_loss == pytest.approx(-52.0)

    def test_no_round_trips_returns_zero_not_an_error(self):
        assert win_rate([]) == 0.0
        assert average_win_loss([]) == (0.0, 0.0)

    def test_all_losing_trades(self):
        trades = [
            fill(ts(0), "BUY", 10, 100.0, commission=0.0),
            fill(ts(1), "SELL", 10, 90.0, commission=0.0),
        ]
        round_trips = extract_round_trips(trades)
        assert win_rate(round_trips) == 0.0
        avg_win, avg_loss = average_win_loss(round_trips)
        assert avg_win == 0.0
        assert avg_loss == pytest.approx(-100.0)


class TestGenerateAndFormatTearsheet:
    def test_tearsheet_dict_matches_individual_metrics(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1010.0), snap(ts(2), 1040.3)]
        trades = [
            fill(ts(0), "BUY", 10, 100.0, commission=1.0),
            fill(ts(1), "SELL", 10, 110.0, commission=1.0),
        ]

        sheet = generate_tearsheet(curve, trades)

        assert sheet["total_return"] == pytest.approx(total_return(curve))
        assert sheet["sharpe_ratio"] == pytest.approx(sharpe_ratio(curve))
        assert sheet["num_round_trips"] == 1
        assert sheet["win_rate"] == pytest.approx(1.0)

    def test_format_tearsheet_includes_title_and_key_labels(self):
        curve = [snap(ts(0), 1000.0), snap(ts(1), 1010.0)]
        text = format_tearsheet(generate_tearsheet(curve, []), title="My Universe")

        assert "My Universe" in text
        assert "Sharpe ratio" in text
        assert "Max drawdown" in text
        assert "Win rate" in text

    def test_raises_on_single_point_curve(self):
        with pytest.raises(PerformanceError):
            generate_tearsheet([snap(ts(0), 1000.0)], [])
