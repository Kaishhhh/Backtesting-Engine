"""Performance analytics: turns a Portfolio's equity curve and trade log
into standard risk/return metrics and a plain-text tearsheet.

Deliberately analytical, not visual -- no plotting library dependency (see
CLAUDE.md's build order: "a simple text/dict tearsheet, not a full HTML
report"). Every annualization/risk-free-rate assumption below is a named
constant, not a magic number buried in a formula.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import sqrt

from engine.events import FillEvent
from engine.portfolio import PortfolioSnapshot

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25  # includes leap years, for CAGR's elapsed-time calc
DEFAULT_RISK_FREE_RATE = 0.0

# Sharpe/Sortino's denominator (volatility / downside deviation) is undefined
# when there's too little data to measure it (fewer than 2 return periods)
# or when it's genuinely zero (a flat curve, or -- for Sortino -- a curve
# with no losing periods at all). Returning this constant instead of raising
# or producing inf/NaN treats "genuinely flat" and "not enough data to tell"
# the same honest way: as "no measurable risk-adjusted edge in this data",
# rather than claiming an answer the data can't support.
UNDEFINED_RATIO_FALLBACK = 0.0


class PerformanceError(RuntimeError):
    """Raised when a metric is asked a question its input can't answer
    (fewer than 2 equity-curve points, a starting value of 0, a SELL fill
    with no matching open BUY lot)."""


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown_pct: float  # negative fraction, e.g. -0.25 for a 25% decline
    peak_timestamp: datetime
    trough_timestamp: datetime
    duration_days: int  # peak-to-trough (the decline phase), NOT peak-to-recovery
    recovered: bool
    recovery_timestamp: datetime | None


@dataclass(frozen=True)
class RoundTrip:
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float  # nets out BOTH entry and exit commission -- see extract_round_trips


@dataclass
class _OpenLot:
    quantity: int
    price: float
    commission_per_share: float
    timestamp: datetime


def _values(equity_curve: list[PortfolioSnapshot]) -> list[float]:
    return [snap.total_value for snap in equity_curve]


def _require_min_points(equity_curve: list[PortfolioSnapshot], minimum: int = 2) -> None:
    if len(equity_curve) < minimum:
        raise PerformanceError(
            f"need at least {minimum} equity-curve points, got {len(equity_curve)}"
        )


def periodic_returns(equity_curve: list[PortfolioSnapshot]) -> list[float]:
    """Simple (not log) returns between consecutive equity-curve snapshots."""
    _require_min_points(equity_curve)
    values = _values(equity_curve)
    return [values[i] / values[i - 1] - 1 for i in range(1, len(values))]


def total_return(equity_curve: list[PortfolioSnapshot]) -> float:
    _require_min_points(equity_curve)
    start, end = equity_curve[0].total_value, equity_curve[-1].total_value
    if start == 0:
        raise PerformanceError("cannot compute a return from a starting value of 0")
    return end / start - 1


def cagr(
    equity_curve: list[PortfolioSnapshot],
    calendar_days_per_year: float = CALENDAR_DAYS_PER_YEAR,
) -> float:
    _require_min_points(equity_curve)
    start_snap, end_snap = equity_curve[0], equity_curve[-1]
    if start_snap.total_value == 0:
        raise PerformanceError("cannot compute CAGR from a starting value of 0")
    # total_seconds(), not .days -- .days truncates any sub-day remainder,
    # which would silently bias a multi-year CAGR calculation.
    seconds = (end_snap.timestamp - start_snap.timestamp).total_seconds()
    years = seconds / (calendar_days_per_year * 86_400)
    if years <= 0:
        raise PerformanceError("cannot compute CAGR over a zero-or-negative-length period")
    growth = end_snap.total_value / start_snap.total_value
    return growth ** (1 / years) - 1


def _annualized_ratio(excess_returns: list[float], deviation: float, trading_days_per_year: int) -> float:
    if len(excess_returns) < 2 or deviation == 0:
        return UNDEFINED_RATIO_FALLBACK
    return statistics.mean(excess_returns) / deviation * sqrt(trading_days_per_year)


def sharpe_ratio(
    equity_curve: list[PortfolioSnapshot],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    _require_min_points(equity_curve)
    returns = periodic_returns(equity_curve)
    per_period_rf = risk_free_rate / trading_days_per_year
    excess = [r - per_period_rf for r in returns]
    deviation = statistics.stdev(excess) if len(excess) >= 2 else 0.0
    return _annualized_ratio(excess, deviation, trading_days_per_year)


def sortino_ratio(
    equity_curve: list[PortfolioSnapshot],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    _require_min_points(equity_curve)
    returns = periodic_returns(equity_curve)
    per_period_rf = risk_free_rate / trading_days_per_year
    excess = [r - per_period_rf for r in returns]
    # Downside deviation over ALL periods, not just losing ones -- the
    # standard Sortino/Van der Meer convention. Up-periods contribute 0, so
    # more winning periods still (correctly) pulls this toward 0, not away.
    downside = [min(r, 0.0) for r in excess]
    deviation = sqrt(statistics.mean(d * d for d in downside)) if downside else 0.0
    return _annualized_ratio(excess, deviation, trading_days_per_year)


def max_drawdown(equity_curve: list[PortfolioSnapshot]) -> DrawdownResult:
    _require_min_points(equity_curve)

    peak_value = equity_curve[0].total_value
    peak_timestamp = equity_curve[0].timestamp
    worst_pct = 0.0
    worst_peak_value = peak_value
    worst_peak_timestamp = peak_timestamp
    worst_trough_timestamp = peak_timestamp

    for snap in equity_curve:
        if snap.total_value > peak_value:
            peak_value = snap.total_value
            peak_timestamp = snap.timestamp
        drawdown_pct = snap.total_value / peak_value - 1
        if drawdown_pct < worst_pct:
            worst_pct = drawdown_pct
            worst_peak_value = peak_value
            worst_peak_timestamp = peak_timestamp
            worst_trough_timestamp = snap.timestamp

    recovered = False
    recovery_timestamp: datetime | None = None
    for snap in equity_curve:
        if snap.timestamp > worst_trough_timestamp and snap.total_value >= worst_peak_value:
            recovered = True
            recovery_timestamp = snap.timestamp
            break

    return DrawdownResult(
        max_drawdown_pct=worst_pct,
        peak_timestamp=worst_peak_timestamp,
        trough_timestamp=worst_trough_timestamp,
        duration_days=(worst_trough_timestamp - worst_peak_timestamp).days,
        recovered=recovered,
        recovery_timestamp=recovery_timestamp,
    )


def extract_round_trips(trade_log: list[FillEvent]) -> list[RoundTrip]:
    """FIFO-match BUY fills to SELL fills, per symbol, into realized round
    trips.

    General lot-matching, not specific to any one strategy's buy/sell
    pattern: a SELL that only partially consumes the oldest open lot splits
    at that lot's boundary, and a SELL spanning several lots produces one
    RoundTrip per lot consumed. Each RoundTrip's pnl nets out BOTH the
    entry and exit fill's commission (true trade economics) -- this is
    deliberately more conservative than summing Portfolio.positions[...]
    .realized_pnl, whose running total only nets the *sell-side* commission
    (buy-side commission already reduced cash at entry without being logged
    against "realized" PnL there). Both figures are correct; they answer
    different questions.
    """
    open_lots: dict[str, deque[_OpenLot]] = {}
    round_trips: list[RoundTrip] = []

    for fill in trade_log:
        lots = open_lots.setdefault(fill.symbol, deque())

        if fill.direction == "BUY":
            lots.append(
                _OpenLot(
                    quantity=fill.quantity,
                    price=fill.fill_price,
                    commission_per_share=fill.commission / fill.quantity,
                    timestamp=fill.timestamp,
                )
            )
            continue

        remaining = fill.quantity
        exit_commission_per_share = fill.commission / fill.quantity
        while remaining > 0:
            if not lots:
                raise PerformanceError(
                    f"SELL fill for {fill.quantity} {fill.symbol} @ {fill.timestamp} "
                    f"has no matching open BUY lot -- trade_log is inconsistent."
                )
            lot = lots[0]
            matched = min(lot.quantity, remaining)
            entry_commission = matched * lot.commission_per_share
            exit_commission = matched * exit_commission_per_share
            pnl = (fill.fill_price - lot.price) * matched - entry_commission - exit_commission
            round_trips.append(
                RoundTrip(
                    symbol=fill.symbol,
                    entry_timestamp=lot.timestamp,
                    exit_timestamp=fill.timestamp,
                    quantity=matched,
                    entry_price=lot.price,
                    exit_price=fill.fill_price,
                    pnl=pnl,
                )
            )
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity == 0:
                lots.popleft()

    return round_trips


def win_rate(round_trips: list[RoundTrip]) -> float:
    """Fraction of round trips with positive pnl. 0.0 for no round trips at
    all -- a strategy that never traded is a normal outcome, not an error.
    Breakeven trades (pnl == 0) count toward the denominator but are wins
    for neither this nor average_win_loss."""
    if not round_trips:
        return 0.0
    wins = sum(1 for rt in round_trips if rt.pnl > 0)
    return wins / len(round_trips)


def average_win_loss(round_trips: list[RoundTrip]) -> tuple[float, float]:
    """(average winning pnl, average losing pnl). 0.0 for either side with
    no qualifying round trips."""
    wins = [rt.pnl for rt in round_trips if rt.pnl > 0]
    losses = [rt.pnl for rt in round_trips if rt.pnl < 0]
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    return avg_win, avg_loss


def generate_tearsheet(
    equity_curve: list[PortfolioSnapshot],
    trade_log: list[FillEvent],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    calendar_days_per_year: float = CALENDAR_DAYS_PER_YEAR,
) -> dict[str, object]:
    """Assemble every metric above into one summary dict."""
    _require_min_points(equity_curve)
    round_trips = extract_round_trips(trade_log)
    avg_win, avg_loss = average_win_loss(round_trips)
    drawdown = max_drawdown(equity_curve)

    return {
        "start": equity_curve[0].timestamp,
        "end": equity_curve[-1].timestamp,
        "starting_value": equity_curve[0].total_value,
        "ending_value": equity_curve[-1].total_value,
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, calendar_days_per_year),
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate, trading_days_per_year),
        "sortino_ratio": sortino_ratio(equity_curve, risk_free_rate, trading_days_per_year),
        "max_drawdown_pct": drawdown.max_drawdown_pct,
        "max_drawdown_peak": drawdown.peak_timestamp,
        "max_drawdown_trough": drawdown.trough_timestamp,
        "max_drawdown_duration_days": drawdown.duration_days,
        "max_drawdown_recovered": drawdown.recovered,
        "num_round_trips": len(round_trips),
        "win_rate": win_rate(round_trips),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def format_tearsheet(sheet: dict[str, object], title: str | None = None) -> str:
    """Render a generate_tearsheet() dict as plain aligned text -- no
    plotting library, per this module's docstring."""
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(f"Period:              {sheet['start']:%Y-%m-%d} to {sheet['end']:%Y-%m-%d}")
    lines.append(f"Starting value:      ${sheet['starting_value']:,.2f}")
    lines.append(f"Ending value:        ${sheet['ending_value']:,.2f}")
    lines.append(f"Total return:        {sheet['total_return'] * 100:.2f}%")
    lines.append(f"CAGR:                {sheet['cagr'] * 100:.2f}%")
    lines.append(f"Sharpe ratio:        {sheet['sharpe_ratio']:.2f}")
    lines.append(f"Sortino ratio:       {sheet['sortino_ratio']:.2f}")
    recovered_note = "recovered" if sheet["max_drawdown_recovered"] else "not recovered by end of period"
    lines.append(
        f"Max drawdown:        {sheet['max_drawdown_pct'] * 100:.2f}% "
        f"(peak {sheet['max_drawdown_peak']:%Y-%m-%d} -> "
        f"trough {sheet['max_drawdown_trough']:%Y-%m-%d}, "
        f"{sheet['max_drawdown_duration_days']}d decline, {recovered_note})"
    )
    lines.append(f"Round trips:         {sheet['num_round_trips']}")
    lines.append(f"Win rate:            {sheet['win_rate'] * 100:.1f}%")
    lines.append(f"Avg win / avg loss:  ${sheet['avg_win']:,.2f} / ${sheet['avg_loss']:,.2f}")
    return "\n".join(lines)
