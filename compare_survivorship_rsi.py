"""Survivorship-bias comparison for RSIMeanReversionStrategy -- the same
20-ticker pool, universe toggle, and aggregation/forward-fill logic as
compare_survivorship.py (see that module's docstring for the full
rationale), just with a different strategy_factory/label passed into its
main(). Confirms the comparison generalizes to a second, conceptually
different strategy rather than being SMA-crossover-specific.
"""

from __future__ import annotations

from compare_survivorship import main
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0

if __name__ == "__main__":
    main(
        strategy_factory=lambda: RSIMeanReversionStrategy(
            period=PERIOD, oversold=OVERSOLD, overbought=OVERBOUGHT
        ),
        label=f"RSI({PERIOD}) mean-reversion ({OVERSOLD:.0f}/{OVERBOUGHT:.0f})",
    )
