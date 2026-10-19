"""Leak-free version of features.py for the daily cross-sectional stock forecast.

Inputs (same as features.py)
  prices:        DataFrame, index = trading dates, columns = tickers, values = close prices
                 (NaN = no print / not listed)
  fundamentals:  long DataFrame with columns ticker, period_end, filing_date, eps

Output (same columns as features.py)
  long DataFrame: date, ticker, mom_z, rev_pct, vol, weekly_mom, cs_move, high_vol, eps, ey

Timing contract
  Every feature in the row for date t is computed only from data available at the close of
  t-1 (project rule: shift every feature by one period). So the row for date t is safe to pair
  with the scored return close t -> close t+1, and it is also safe if the label is accidentally
  aligned as the same-day return close t-1 -> close t. For the live submission after close t,
  use the last row of build_features(prices up to t, fundamentals): the backtest and the live
  submission use the same code path and the same row.

Availability declared per input (the reference time is not the availability time)
  close price of day d     : available at the close of d
  eps for period_end Q     : available from filing_date + 1 calendar day (the filing may land
                             after the close), never from period_end
  derived features         : available at the latest availability among their inputs

See audit_report.md for the problems found in features.py and how each fix was checked.
"""
import numpy as np
import pandas as pd

FEATURE_LAG = 1                        # trading days every feature is shifted (project rule)
FFILL_LIMIT = 5                        # carry the last known close over a missing print, at most 5 days
ZSCORE_MIN_PERIODS = 60                # warm-up for the expanding (past-only) z-score of momentum
RANK_MIN_PERIODS = 60                  # warm-up for the expanding (past-only) percentile of reversal
REGIME_MIN_PERIODS = 120               # warm-up for the expanding 80th percentile of market vol
FILING_LAG = pd.Timedelta(days=1)      # eps becomes usable the day after its filing_date

FEATURE_COLUMNS = ["mom_z", "rev_pct", "vol", "weekly_mom", "cs_move", "high_vol", "eps", "ey"]


# --- step 1: price panel ---------------------------------------------------------------
def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep every ticker (no survivorship filter) and fill gaps only from the past."""
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    if not px.index.is_monotonic_increasing or px.index.has_duplicates:
        raise ValueError("prices index must be sorted, unique trading dates")
    return px.ffill(limit=FFILL_LIMIT)


def _returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change(fill_method=None)


# --- step 2: one function per feature, each uses rows <= t only ------------------------
def _mom_z(px: pd.DataFrame) -> pd.DataFrame:
    """20-day momentum, z-scored per ticker against its own past (expanding window)."""
    mom = px.pct_change(20, fill_method=None)
    past_mean = mom.expanding(min_periods=ZSCORE_MIN_PERIODS).mean()
    past_std = mom.expanding(min_periods=ZSCORE_MIN_PERIODS).std()
    return (mom - past_mean) / past_std


def _rev_pct(ret: pd.DataFrame) -> pd.DataFrame:
    """5-day reversal as a percentile of the ticker's own history up to t (expanding rank)."""
    rev = -ret.rolling(5).sum()
    return rev.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True)


def _vol(ret: pd.DataFrame) -> pd.DataFrame:
    """21-day trailing volatility (window ends at t, not centred on t)."""
    return ret.rolling(21).std()


def _high_vol(ret: pd.DataFrame) -> pd.Series:
    """1 when 20-day market vol is above the 80th percentile of market vol seen so far."""
    mkt_vol = ret.mean(axis=1).rolling(20).std()
    threshold = mkt_vol.expanding(min_periods=REGIME_MIN_PERIODS).quantile(0.8)
    flag = (mkt_vol > threshold).astype(float)
    return flag.where(mkt_vol.notna() & threshold.notna())       # NaN during warm-up


def _weekly_mom(px: pd.DataFrame) -> pd.DataFrame:
    """4-week momentum on Friday closes, carried forward to the following days only."""
    weekly = px.resample("W-FRI", label="right", closed="right").last()
    weekly = weekly.pct_change(4, fill_method=None)
    return weekly.reindex(px.index, method="ffill")


def _cs_move(ret: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of the day's return (across tickers on the same date)."""
    return ret.rank(axis=1, pct=True) - 0.5


def _eps_panel(fundamentals: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Latest eps whose filing was available on each date (point-in-time join on availability)."""
    f = fundamentals[["ticker", "period_end", "filing_date", "eps"]].copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["available_at"] = pd.to_datetime(f["filing_date"]) + FILING_LAG
    f = f.sort_values(["available_at", "period_end"])          # same-day ties: newest period wins
    grid = px.stack(future_stack=True).rename("px").reset_index()
    grid.columns = ["date", "ticker", "px"]
    grid = grid.sort_values("date")
    joined = pd.merge_asof(grid, f[["ticker", "available_at", "eps"]], left_on="date",
                           right_on="available_at", by="ticker", direction="backward")
    eps = joined.pivot(index="date", columns="ticker", values="eps")
    return eps.reindex(index=px.index, columns=px.columns)


# --- step 3: assemble, apply the one-period shift, and stack ---------------------------
def _feature_panels(prices: pd.DataFrame, fundamentals: pd.DataFrame, lag: int) -> dict:
    px = _clean_prices(prices)
    ret = _returns(px)
    eps = _eps_panel(fundamentals, px)
    raw = {
        "mom_z": _mom_z(px),
        "rev_pct": _rev_pct(ret),
        "vol": _vol(ret),
        "weekly_mom": _weekly_mom(px),
        "cs_move": _cs_move(ret),
        "eps": eps,
        "ey": eps / px,
    }
    shifted = {name: panel.shift(lag) for name, panel in raw.items()}
    shifted["high_vol"] = _high_vol(ret).shift(lag)
    return shifted


def _assemble(prices: pd.DataFrame, fundamentals: pd.DataFrame, lag: int = FEATURE_LAG) -> pd.DataFrame:
    panels = _feature_panels(prices, fundamentals, lag)
    high_vol = panels.pop("high_vol")
    long = pd.DataFrame({name: df.stack(future_stack=True) for name, df in panels.items()})
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    long["high_vol"] = long["date"].map(high_vol)
    long = long[["date", "ticker"] + FEATURE_COLUMNS]
    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Row for date t uses only data available at the close of t-1 (see module docstring)."""
    return _assemble(prices, fundamentals, lag=FEATURE_LAG)
