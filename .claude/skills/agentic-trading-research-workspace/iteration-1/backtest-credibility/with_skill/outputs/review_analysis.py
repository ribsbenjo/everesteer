#!/usr/bin/env python3
"""Numbers behind review.md. Read-only on the input file.

Run:  python3 review_analysis.py > review_analysis_output.txt
Cells are marked with `# %%` so the file opens as a notebook in VS Code / Jupytext.
"""

# %% Cell 1: imports and paths
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]  # .../.claude/skills
INPUT = ROOT / "agentic-trading-research-workspace/inputs/backtest-review/returns.csv"
sys.path.insert(0, str(ROOT / "agentic-trading-research/scripts"))
import robust_stats as rs  # noqa: E402

pd.set_option("display.width", 140)
PPY = 252
ANN = np.sqrt(PPY)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# %% Cell 2: load the file exactly as delivered
df = pd.read_csv(INPUT, index_col=0, parse_dates=True).sort_index()
df["active"] = df["strategy"] - df["benchmark"]
section("1. File as delivered")
print(f"rows={len(df)}  first={df.index.min().date()}  last={df.index.max().date()}  NaNs={int(df.isna().sum().sum())}")
print("weekday counts (Mon=0):", df.index.dayofweek.value_counts().sort_index().to_dict())
print("calendar years covered: 2  ->  rows per year in file:", len(df) / 2)

# %% Cell 3: calendar integrity - NYSE full-day closures 2023-2024
NYSE_CLOSED = pd.to_datetime([
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04",
    "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
])
on_holiday = df.index.isin(NYSE_CLOSED)
section("2. Rows dated on NYSE full-day market closures")
print(f"{on_holiday.sum()} rows fall on days the exchange was closed; all have non-zero returns:")
print(df.loc[on_holiday, ["strategy", "benchmark"]].round(4).to_string())
print("NYSE trading days in window (2023: 250, 2024: 252) = 502; file has", len(df))
print("largest |benchmark| day in the file:", df["benchmark"].abs().idxmax().date(),
      f"{df.loc[df['benchmark'].abs().idxmax(), 'benchmark']:+.4f}")

# %% Cell 4: spot-check the benchmark against well-known market days
section("3. Benchmark on well-known market days (S&P 500 moves from public record, approximate)")
known = {
    "2023-03-09": "SVB stress, S&P 500 about -1.8%",
    "2024-08-05": "yen carry unwind / VIX spike, S&P 500 about -3.0%",
    "2024-12-18": "hawkish FOMC, S&P 500 about -2.9%",
}
for d, note in known.items():
    print(f"{d}  benchmark={df.loc[d, 'benchmark']:+.4f}  strategy={df.loc[d, 'strategy']:+.4f}   ({note})")

# %% Cell 5: headline metrics as delivered (all 521 rows, sqrt(252))
section("4. Headline metrics, file as delivered (521 rows, annualised with 252)")
rows = {}
for c in ["strategy", "benchmark", "active"]:
    s = rs.summary(df[c], PPY, block=10)
    rows[c] = dict(n=s["n"], ann_mean=s["mean"] * PPY, ann_vol=s["vol"] * ANN, sharpe=s["sharpe"],
                   total=(1 + df[c]).prod() - 1, mdd=s["max_drawdown"], cvar95=s["cvar95"],
                   cvar99=s["cvar99"], skew=s["skew"], kurt_raw=s["kurt"],
                   mean_ci_ann_lo=s["mean_ci95"][0] * PPY, mean_ci_ann_hi=s["mean_ci95"][1] * PPY)
print(pd.DataFrame(rows).T.round(4).to_string())
cagr = {c: (1 + df[c]).prod() ** 0.5 - 1 for c in ["strategy", "benchmark"]}
print("CAGR over 2 calendar years:", {k: round(v, 4) for k, v in cagr.items()})
RF = 0.05  # assumed flat T-bill yield for 2023-2024 (3m bills were roughly 4.3-5.5% over the window)
print(f"Sharpe net of an assumed {RF:.0%} risk-free rate:",
      {c: round((df[c].mean() * PPY - RF) / (df[c].std() * ANN), 3) for c in ["strategy", "benchmark"]})

# %% Cell 6: same metrics after dropping the 19 holiday rows (what a real calendar would give)
section("5. Same metrics on the 502 exchange trading days only")
tdf = df.loc[~on_holiday]
for c in ["strategy", "benchmark", "active"]:
    print(f"{c:>9}: Sharpe/IR={rs.sharpe(tdf[c], PPY):.3f}  total={(1 + tdf[c]).prod() - 1:+.4f}")

# %% Cell 7: CAPM decomposition with Newey-West (HAC, 5 lags) t-stats
section("6. Regression on the benchmark (Newey-West HAC, 5 lags)")
X = sm.add_constant(df["benchmark"])
capm = sm.OLS(df["strategy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"alpha/day={capm.params['const']:.5f} (ann {capm.params['const'] * PPY:.3f}), t={capm.tvalues['const']:.2f}")
print(f"beta={capm.params['benchmark']:.3f}, t={capm.tvalues['benchmark']:.2f}, R2={capm.rsquared:.3f}, "
      f"corr={df['strategy'].corr(df['benchmark']):.3f}")
resid = capm.resid
print(f"residual vol ann={resid.std() * ANN:.3f}; appraisal ratio (alpha/resid vol) ann="
      f"{capm.params['const'] / resid.std() * ANN:.3f}")
Xtm = sm.add_constant(pd.DataFrame({"b": df["benchmark"], "b2": df["benchmark"] ** 2}))
tm = sm.OLS(df["strategy"], Xtm).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"Treynor-Mazuy timing gamma={tm.params['b2']:.2f}, t={tm.tvalues['b2']:.2f}; alpha t={tm.tvalues['const']:.2f}")
up = df["benchmark"] > 0
print(f"up-market capture={df.loc[up, 'strategy'].mean() / df.loc[up, 'benchmark'].mean():.3f}  "
      f"down-market capture={df.loc[~up, 'strategy'].mean() / df.loc[~up, 'benchmark'].mean():.3f}")
print(f"hit rate: strategy {(df['strategy'] > 0).mean():.3f}, benchmark {(df['benchmark'] > 0).mean():.3f}, "
      f"active>0 {(df['active'] > 0).mean():.3f}")
print("autocorr lag1:", {c: round(df[c].autocorr(1), 3) for c in ["strategy", "benchmark", "active"]})

# %% Cell 8: sub-periods (years, halves, GPT-4o knowledge cutoff Oct 2023)
section("7. Sub-periods")


def period_row(g):
    return pd.Series(dict(n=len(g), strat_sharpe=rs.sharpe(g["strategy"], PPY), bench_sharpe=rs.sharpe(g["benchmark"], PPY),
                          strat_ret=(1 + g["strategy"]).prod() - 1, bench_ret=(1 + g["benchmark"]).prod() - 1,
                          active_IR=rs.sharpe(g["active"], PPY)))


print(df.groupby(df.index.year).apply(period_row).round(3).to_string())
halves = np.where(df.index.month <= 6, "H1", "H2")
print(df.groupby([df.index.year, halves]).apply(period_row).round(3).to_string())
cut = pd.Timestamp("2023-09-30")  # GPT-4o snapshots: knowledge cutoff 1 Oct 2023 (assumed)
print("\nSplit at the assumed GPT-4o training cutoff (Oct 2023):")
print(pd.DataFrame({"pre-cutoff (<=2023-09-30)": period_row(df.loc[:cut]),
                    "post-cutoff (>=2023-10-01)": period_row(df.loc[cut + pd.Timedelta(days=1):])}).T.round(3).to_string())
q = df.groupby(df.index.to_period("Q")).apply(period_row)
print("\nQuarterly:")
print(q.round(3).to_string())
print(f"quarters with active return > 0: {(q['strat_ret'] > q['bench_ret']).sum()} of {len(q)}")

# %% Cell 9: moving-block bootstrap CIs (Sharpe, Sharpe difference, mean active return)
section("8. Moving-block bootstrap (circular, 10,000 draws)")


def joint_block_bootstrap(arr, stat, block, n_boot=10_000, seed=0):
    """Resample rows of a 2-D array with the same circular blocks so pairs stay aligned."""
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    return np.array([stat(arr[i]) for i in idx])


def sr(x):
    return x.mean(axis=0) / x.std(axis=0, ddof=1) * ANN


arr = df[["strategy", "benchmark"]].to_numpy()
for block in (5, 10, 20):
    d = joint_block_bootstrap(arr, sr, block)
    diff = d[:, 0] - d[:, 1]
    pc = rs.paired_block_bootstrap_ci(df["strategy"], df["benchmark"], block=block, n_boot=10_000)
    print(f"block={block:>2}: strategy SR 95% CI [{np.quantile(d[:, 0], .025):.2f}, {np.quantile(d[:, 0], .975):.2f}]; "
          f"SR(strat)-SR(bench) CI [{np.quantile(diff, .025):.2f}, {np.quantile(diff, .975):.2f}], "
          f"P(diff<=0)={(diff <= 0).mean():.4f}; "
          f"mean active ann CI [{pc['ci'][0] * PPY:.3f}, {pc['ci'][1] * PPY:.3f}]")
w = rs.wilcoxon_signed_rank(df["strategy"].to_numpy(), df["benchmark"].to_numpy())
print(f"Wilcoxon signed-rank strategy vs benchmark (secondary, assumes independence): z={w['z']:.2f}, p={w['p']:.2g}")
se_lo = np.sqrt((1 + 0.5 * (rs.sharpe(df['strategy']) ** 2)) / (len(df) - 1)) * ANN
print(f"Analytic (Lo 2002, iid) SE of annualised Sharpe at n={len(df)}: {se_lo:.2f}")

# %% Cell 10: deflated Sharpe ratio for the 40-variant search
section("9. Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) - search budget 40 prompt variants")
n = len(df)
for label, col in [("strategy Sharpe vs 0", "strategy"), ("active IR vs benchmark (stock-picking skill)", "active")]:
    x = df[col]
    sr_pp = rs.sharpe(x)
    print(f"\n{label}: per-period SR={sr_pp:.4f} (ann {sr_pp * ANN:.2f}), skew={x.skew():.3f}, raw kurt={x.kurt() + 3:.3f}, "
          f"PSR vs 0 = {rs.probabilistic_sharpe_ratio(sr_pp, n, x.skew(), x.kurt() + 3):.4f}")
    grid = {}
    for sd_ann in (0.25, 0.50, 0.75, 1.00):
        var_pp = (sd_ann / ANN) ** 2
        grid[f"sd(SR) across variants={sd_ann:.2f}"] = {
            f"N_eff={k}": "SR0={:.2f} DSR={:.3f}".format(
                rs.expected_max_sharpe(k, var_pp) * ANN,
                rs.deflated_sharpe_ratio(sr_pp, n, x.skew(), x.kurt() + 3, k, var_pp)["dsr"])
            for k in (5, 10, 20, 40)}
    print(pd.DataFrame(grid).T.to_string())

# %% Cell 11: transaction-cost sensitivity (positions not provided, so turnover is a scenario)
section("10. Cost sensitivity: annualised Sharpe after costs (turnover scenarios; positions not in file)")
mu, vol = df["strategy"].mean() * PPY, df["strategy"].std() * ANN
amu = df["active"].mean() * PPY
tbl = {}
for to in (0.10, 0.25, 0.50, 1.00):  # daily two-sided turnover as fraction of NAV
    tbl[f"daily turnover {to:.0%}"] = {f"{bps} bps/side": round((mu - to * PPY * bps / 1e4) / vol, 2) for bps in (2, 5, 10, 20)}
print(pd.DataFrame(tbl).T.to_string())
for target, lab in [(0.0, "Sharpe 0"), (rs.sharpe(df["benchmark"], PPY), "benchmark Sharpe")]:
    be = (mu - target * vol) / PPY  # daily cost budget as fraction of NAV
    print(f"daily cost budget before strategy falls to {lab}: {be * 1e4:.1f} bps of NAV/day "
          f"(= {be * 1e4 / 0.5:.1f} bps/side at 50% daily turnover, {be * 1e4 / 1.0:.1f} bps/side at 100%)")
print(f"annual excess return over benchmark to absorb costs: {amu:.3f}")

# %% Cell 12: how much of the result sits on the 19 phantom (holiday) rows
section("11. Contribution of the 19 holiday rows, and key tests re-run on the 502 trading days")
h = df.loc[on_holiday]
print(f"sum of returns on holiday rows: strategy {h['strategy'].sum():+.4f}, benchmark {h['benchmark'].sum():+.4f}, "
      f"active {h['active'].sum():+.4f}  (vs full-file active sum {df['active'].sum():+.4f})")
print(f"holiday rows = {on_holiday.mean():.1%} of rows but {h['active'].sum() / df['active'].sum():.1%} of summed active return")
capm_t = sm.OLS(tdf["strategy"], sm.add_constant(tdf["benchmark"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"trading days only: alpha ann={capm_t.params['const'] * PPY:.3f} (HAC t={capm_t.tvalues['const']:.2f}), "
      f"beta={capm_t.params['benchmark']:.3f}")
arr_t = tdf[["strategy", "benchmark"]].to_numpy()
d = joint_block_bootstrap(arr_t, sr, 10)
diff = d[:, 0] - d[:, 1]
pc = rs.paired_block_bootstrap_ci(tdf["strategy"], tdf["benchmark"], block=10, n_boot=10_000)
print(f"trading days only, block=10: strategy SR CI [{np.quantile(d[:, 0], .025):.2f}, {np.quantile(d[:, 0], .975):.2f}]; "
      f"SR diff CI [{np.quantile(diff, .025):.2f}, {np.quantile(diff, .975):.2f}], P(diff<=0)={(diff <= 0).mean():.4f}; "
      f"mean active ann CI [{pc['ci'][0] * PPY:.3f}, {pc['ci'][1] * PPY:.3f}]")
x = tdf["active"]
sr_pp = rs.sharpe(x)
print(f"trading days only, active IR={sr_pp * ANN:.2f}, PSR vs 0={rs.probabilistic_sharpe_ratio(sr_pp, len(x), x.skew(), x.kurt() + 3):.3f}")
for sd_ann in (0.25, 0.50, 0.75):
    print(f"  DSR (N_eff=40, sd(SR)={sd_ann}): "
          f"{rs.deflated_sharpe_ratio(sr_pp, len(x), x.skew(), x.kurt() + 3, 40, (sd_ann / ANN) ** 2)['dsr']:.3f};  "
          f"(N_eff=10): {rs.deflated_sharpe_ratio(sr_pp, len(x), x.skew(), x.kurt() + 3, 10, (sd_ann / ANN) ** 2)['dsr']:.3f}")
print("max drawdown (strategy, benchmark):", round(rs.max_drawdown(df['strategy']), 4), round(rs.max_drawdown(df['benchmark']), 4))

# %% Cell 13: inference-cost scenario (assumptions stated, verify against invoices)
section("12. Inference cost scenario (assumptions, not data)")
tickers, days, calls, in_tok, out_tok = 500, 502, 12, 6_000, 600
price_in, price_out = 2.50, 10.00  # USD per 1M tokens, GPT-4o list price (check current pricing)
per_call = in_tok / 1e6 * price_in + out_tok / 1e6 * price_out
one_run = tickers * days * calls * per_call
print(f"per call ${per_call:.4f}; one full backtest run ${one_run:,.0f}; 40 variants ${40 * one_run:,.0f}; "
      f"live per year ${tickers * PPY * calls * per_call:,.0f}")
