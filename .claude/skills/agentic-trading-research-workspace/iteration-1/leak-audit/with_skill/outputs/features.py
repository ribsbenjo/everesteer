"""Features for our daily cross-sectional stock forecast.

prices:        DataFrame, index = trading dates, columns = tickers, values = close prices (NaN = no print / not listed)
fundamentals:  long DataFrame with columns ticker, period_end, filing_date, eps
build_features returns a long DataFrame (date, ticker, feature columns) that the model trains on.
"""
import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    # clean up the price panel
    prices = prices.dropna(axis=1)            # keep tickers with a full history
    prices = prices.bfill()                   # fill the odd missing print

    ret = prices.pct_change()

    # 1) 20-day momentum, normalised so features are comparable over time
    mom = prices.pct_change(20)
    mom_z = (mom - mom.mean()) / mom.std()

    # 2) short-term reversal, as a percentile
    rev = -ret.rolling(5).sum()
    rev_pct = rev.rank(pct=True)

    # 3) smoothed volatility
    vol = ret.rolling(21, center=True).std()

    # 4) market regime flag: high-vol regime = top 20% of market volatility
    mkt_vol = ret.mean(axis=1).rolling(20).std()
    high_vol = (mkt_vol > mkt_vol.quantile(0.8)).astype(int)

    # 5) slower weekly momentum, mapped back to daily rows
    weekly = prices.resample("W-FRI").last().pct_change(4)
    weekly_mom = weekly.reindex(prices.index, method="bfill")

    # 6) cross-sectional size of the move today (fine: across tickers on the same day)
    cs_move = ret.rank(axis=1, pct=True) - 0.5

    panels = {"mom_z": mom_z, "rev_pct": rev_pct, "vol": vol, "weekly_mom": weekly_mom, "cs_move": cs_move}
    long = pd.DataFrame({name: df.stack(future_stack=True) for name, df in panels.items()})
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    long["high_vol"] = long["date"].map(high_vol)

    # 7) earnings yield from fundamentals
    f = fundamentals.sort_values("period_end")
    long = long.sort_values("date")
    long = pd.merge_asof(long, f[["ticker", "period_end", "eps"]], left_on="date", right_on="period_end",
                         by="ticker", direction="backward")
    px = prices.stack().rename("px").reset_index()
    px.columns = ["date", "ticker", "px"]
    long = long.merge(px, on=["date", "ticker"], how="left")
    long["ey"] = long["eps"] / long["px"]
    return long.drop(columns=["period_end", "px"]).sort_values(["date", "ticker"]).reset_index(drop=True)
