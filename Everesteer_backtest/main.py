"""
main.py: the team's forecasting file. Baseline = plain 20-bar momentum (playbook round 1).

Contract used by backtest.py (keep it when editing):
    signal(prices)   -> date x asset panel, every feature shifted so row t uses data <= t
    forecast(prices) -> pd.Series for the latest row, one value per asset ID

Builder: change ONE thing per round inside signal(). Keep windows round (5, 20, 60).
"""

import pandas as pd


def z(df: pd.DataFrame) -> pd.DataFrame:
    """Rank across assets each row, centred on 0 (range -0.5..0.5)."""
    return df.rank(axis=1, pct=True) - 0.5


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    # The submission at t may use the close at t, so no extra shift here.
    # FILL: if the rules say the bar at t is NOT known when we submit, add .shift(1) to every feature.
    mom = prices.pct_change(20, fill_method=None)
    # reversal = -prices.pct_change(5, fill_method=None)
    # lowvol   = -prices.pct_change(fill_method=None).rolling(60).std()
    return 1.0 * z(mom)  # + 0.5 * z(reversal) + 0.5 * z(lowvol)


def forecast(prices: pd.DataFrame) -> pd.Series:
    sig = signal(prices).iloc[-1]
    # Assets with any history get a neutral 0 if their signal is missing.
    # FILL: replace with the real template's asset list once we have it.
    live = prices.columns[prices.notna().any()]
    shrink = 1.0  # FILL: < 1 if the metric punishes overconfidence
    return (shrink * sig.reindex(live)).fillna(0.0)
