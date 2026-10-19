"""Leak-free features for our daily cross-sectional stock forecast.

Timing convention (the whole file depends on this):
    A row dated t holds only information available at the close of day t.
    We submit after the close of t and are scored on the return close(t) -> close(t+1).
    That target is produced by forward_return() below and is never used as a feature.
    Relative to the scored period, every feature is therefore lagged by one period.
    If your training code instead stores the target on the date the return is realised (t+1),
    shift the features by one row per ticker before joining. Do one or the other, never both or neither.

prices:        DataFrame, index = trading dates, columns = tickers, values = close prices (NaN = no print / not listed)
fundamentals:  long DataFrame with columns ticker, period_end, filing_date, eps
build_features returns a long DataFrame (date, ticker, feature columns) that the model trains on.
Columns and their order are unchanged from features.py.

Assumptions, also listed in audit_report.md:
    * A filing dated D may come out after the close of D, so its EPS is used from the next trading day on.
    * A missing close is carried forward for at most FFILL_LIMIT trading days. Older gaps stay NaN.
    * rev_pct is a cross-sectional percentile (same day, across tickers).
    * mom_z is a per-ticker z-score that uses only that ticker's history up to t (expanding window).
"""
import pandas as pd

FFILL_LIMIT = 5             # max trading days a stale close is carried forward
MOM_Z_MIN_PERIODS = 60      # history needed before the expanding z-score of momentum is defined
REGIME_MIN_PERIODS = 120    # history needed before the expanding 80th-percentile regime threshold is defined
FEATURE_COLUMNS = ["mom_z", "rev_pct", "vol", "weekly_mom", "cs_move", "high_vol", "eps", "ey"]


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep every ticker (no survivorship filter). Fill short gaps with the last known close only."""
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices.ffill(limit=FFILL_LIMIT)


def _momentum_z(px: pd.DataFrame) -> pd.DataFrame:
    """20-day momentum as a z-score against each ticker's own history up to and including t."""
    mom = px.pct_change(20, fill_method=None)
    mu = mom.expanding(min_periods=MOM_Z_MIN_PERIODS).mean()
    sd = mom.expanding(min_periods=MOM_Z_MIN_PERIODS).std()
    return (mom - mu) / sd


def _reversal_pct(ret: pd.DataFrame) -> pd.DataFrame:
    """Minus the 5-day return, as a percentile across tickers on the same day."""
    rev = -ret.rolling(5).sum()
    return rev.rank(axis=1, pct=True)


def _trailing_vol(ret: pd.DataFrame) -> pd.DataFrame:
    """21-day volatility over the window ending at t (trailing, not centred)."""
    return ret.rolling(21).std()


def _high_vol_regime(ret: pd.DataFrame) -> pd.Series:
    """1 if 20-day market volatility is above its 80th percentile so far, 0 if not, NaN while warming up."""
    mkt_vol = ret.mean(axis=1).rolling(20).std()
    threshold = mkt_vol.expanding(min_periods=REGIME_MIN_PERIODS).quantile(0.8)
    flag = (mkt_vol > threshold).astype(float)
    return flag.where(mkt_vol.notna() & threshold.notna())


def _weekly_momentum(px: pd.DataFrame) -> pd.DataFrame:
    """4-week momentum from Friday closes. Each day gets the most recent completed week (forward fill)."""
    weekly = px.resample("W-FRI").last().pct_change(4, fill_method=None)
    return weekly.reindex(px.index, method="ffill")


def _cross_sectional_move(ret: pd.DataFrame) -> pd.DataFrame:
    """Today's return as a percentile across tickers, centred on zero. Known at the close of t."""
    return ret.rank(axis=1, pct=True) - 0.5


def _to_long(panels: dict, px: pd.DataFrame) -> pd.DataFrame:
    """Stack the wide panels into (date, ticker) rows. Keep only tickers that have a usable close at t."""
    long = pd.DataFrame({name: df.stack(future_stack=True) for name, df in panels.items()})
    long["px"] = px.stack(future_stack=True)
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    return long[long["px"].notna()]


def _point_in_time_eps(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """EPS keyed by filing date. A late filing for an older quarter never overwrites a newer quarter."""
    f = fundamentals[["ticker", "period_end", "filing_date", "eps"]].copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    f = f.dropna(subset=["filing_date"]).sort_values(["ticker", "filing_date"])
    newest_before = f.groupby("ticker")["period_end"].cummax().groupby(f["ticker"]).shift(1)
    f = f[newest_before.isna() | (f["period_end"] >= newest_before)]
    return f[["ticker", "filing_date", "eps"]].sort_values("filing_date")


def _add_earnings_yield(long: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest EPS filed strictly before t, then divide by the close at t."""
    f = _point_in_time_eps(fundamentals)
    long = pd.merge_asof(long.sort_values("date"), f, left_on="date", right_on="filing_date",
                         by="ticker", direction="backward", allow_exact_matches=False)
    long["ey"] = long["eps"] / long["px"]
    return long


def build_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    px = _clean_prices(prices)
    ret = px.pct_change(fill_method=None)

    panels = {
        "mom_z": _momentum_z(px),
        "rev_pct": _reversal_pct(ret),
        "vol": _trailing_vol(ret),
        "weekly_mom": _weekly_momentum(px),
        "cs_move": _cross_sectional_move(ret),
    }
    long = _to_long(panels, px)
    long["high_vol"] = long["date"].map(_high_vol_regime(ret))
    long = _add_earnings_yield(long, fundamentals)

    cols = ["date", "ticker"] + FEATURE_COLUMNS
    return long[cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def forward_return(prices: pd.DataFrame) -> pd.DataFrame:
    """The scored target: close(t) -> close(t+1) return, stored on row t. Never feed this to the model as a feature.

    Uses raw closes, so the target is NaN when either day has no print.
    """
    raw = prices.sort_index()
    raw.index = pd.to_datetime(raw.index)
    fwd = raw.shift(-1) / raw - 1
    out = fwd.stack().rename("fwd_ret").reset_index()
    out.columns = ["date", "ticker", "fwd_ret"]
    return out
