"""
make_fake_data.py: synthetic wide price panel so backtest.py can be tested before 18:00.
Delete or ignore once the real dataset is available.

    python make_fake_data.py            # writes data/prices.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_days, n_assets = 750, 50
dates = pd.bdate_range("2023-11-01", periods=n_days)
assets = [f"A{i:03d}" for i in range(n_assets)]

# Daily returns with a weak momentum effect so a momentum signal has some edge.
rets = np.zeros((n_days, n_assets))
for d in range(1, n_days):
    past = rets[max(0, d - 20):d].sum(axis=0)
    rets[d] = 0.02 * (past - past.mean()) / 20 + rng.normal(0, 0.015, n_assets)
prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=assets)

prices.iloc[:400, -1] = np.nan                       # late-listed asset
mask = rng.random(prices.shape) < 0.002              # sporadic missing prints
prices = prices.mask(mask)

out = Path("data/prices.csv")
out.parent.mkdir(exist_ok=True)
prices.rename_axis("date").to_csv(out)
print(f"wrote {out}  {prices.shape[0]} rows x {prices.shape[1]} assets")
