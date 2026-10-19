# %% [markdown]
# Backtest credibility checks for the best GPT-4o multi-agent variant.
# Every number quoted in review.md comes from this script.
# Usage: python3 analysis.py [path/to/returns.csv]
# Read-only on the input. Uses pandas / numpy / scipy only.

# %% Cell 1: imports and constants
import sys
from math import e

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CSV = (
    "/Users/ribizsarbenjamin/Documents/EVK/versenyek/everesteer hackaton/.claude/skills/"
    "agentic-trading-research-workspace/inputs/backtest-review/returns.csv"
)
CSV = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
ANN = 252             # annualisation the team most likely used
RF_ANN = 0.05         # assumption: average 3m T-bill yield 2023-2024 was roughly 5%
N_TRIALS = 40         # prompt variants tried
GPT4O_CUTOFF = "2023-10-01"   # assumption: documented GPT-4o training-data cutoff (Oct 2023)
EULER_GAMMA = 0.5772156649
SEED = 7

# NYSE full-day closures in the sample window
NYSE_HOLIDAYS = pd.to_datetime([
    "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04",
    "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19",
    "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28",
    "2024-12-25",
])


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# %% Cell 2: load data
df = pd.read_csv(CSV, parse_dates=["date"], index_col="date").sort_index()
s, bm = df["strategy"], df["benchmark"]
T = len(df)
years = (df.index[-1] - df.index[0]).days / 365.25
section("1. Data load")
print(f"rows={T}  first={df.index[0].date()}  last={df.index[-1].date()}  years={years:.2f}")
print(f"missing values: {int(df.isna().sum().sum())}   duplicate dates: {int(df.index.duplicated().sum())}")
print(f"rows per calendar year: {df.groupby(df.index.year).size().to_dict()}")


# %% Cell 3: calendar integrity (does the index match NYSE trading days?)
section("2. Calendar integrity")
bdays = pd.bdate_range(df.index[0], df.index[-1])
print(f"index identical to pandas Mon-Fri bdate_range: {df.index.equals(bdays)}")
hol = df.reindex(NYSE_HOLIDAYS)
present = hol.dropna()
print(f"NYSE holidays in window: {len(NYSE_HOLIDAYS)}   rows present on holidays: {len(present)}")
print(f"holiday rows with non-zero strategy return: {(present['strategy'] != 0).sum()}  "
      f"non-zero benchmark return: {(present['benchmark'] != 0).sum()}")
print(f"expected NYSE sessions 2023+2024: 250 + 252 = 502; file has {T}")
print("largest benchmark moves that fall on market holidays:")
print(present.reindex(present["benchmark"].abs().sort_values(ascending=False).index).head(5).round(4))
print(f"rank of 2023-07-04 among worst benchmark days: "
      f"{int(bm.rank().loc['2023-07-04'])} of {T}")
# Spot checks against widely reported market moves (approximate S&P 500 close-to-close moves)
spot = pd.DataFrame({
    "reported_SP500_move": ["about -3.0% (yen carry unwind)", "about -2.9% (Fed day)",
                            "about -1.8% (SVB news)", "about +2.5% (post-election)"],
}, index=pd.to_datetime(["2024-08-05", "2024-12-18", "2023-03-09", "2024-11-06"]))
spot["file_benchmark"] = bm.reindex(spot.index).round(4)
spot["file_strategy"] = s.reindex(spot.index).round(4)
print("spot checks vs reported market moves:")
print(spot)


# %% Cell 4: distribution shape (real daily equity returns are fat-tailed and vol-clustered)
section("3. Distribution shape")


def ljung_box(x, lags=10):
    x = pd.Series(x)
    n = len(x)
    q = n * (n + 2) * sum(x.autocorr(k) ** 2 / (n - k) for k in range(1, lags + 1))
    return q, 1 - stats.chi2.cdf(q, lags)


for name, r in [("strategy", s), ("benchmark", bm)]:
    q_sq, p_sq = ljung_box(r ** 2)
    print(f"{name:9s} skew={stats.skew(r):+.3f}  excess kurtosis={stats.kurtosis(r):+.3f}  "
          f"Jarque-Bera p={stats.jarque_bera(r).pvalue:.3f}  "
          f"Ljung-Box(10) on r^2 p={p_sq:.3f}  lag-1 autocorr={r.autocorr(1):+.3f}")


# %% Cell 5: headline performance at face value
section("4. Headline performance (face value, 252-day annualisation)")


def max_drawdown(r):
    w = (1 + r).cumprod()
    dd = w / w.cummax() - 1
    return dd.min(), dd.idxmin()


def perf(r, rf_ann=0.0, ann=ANN):
    mu, sd = r.mean() * ann, r.std(ddof=1) * np.sqrt(ann)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(ann)
    mdd, mdd_date = max_drawdown(r)
    cagr = (1 + r).prod() ** (1 / years) - 1
    return {
        "total_return": (1 + r).prod() - 1,
        "CAGR": cagr,
        "ann_mean": mu,
        "ann_vol": sd,
        "Sharpe_rf0": mu / sd,
        f"Sharpe_rf{int(RF_ANN*100)}pct": (mu - RF_ANN) / sd,
        "Sortino_rf0": mu / downside,
        "max_drawdown": mdd,
        "Calmar": cagr / abs(mdd),
        "hit_rate_daily": (r > 0).mean(),
    }


headline = pd.DataFrame({"strategy": perf(s), "benchmark": perf(bm)})
print(headline.round(4))
print(f"Sharpe if annualised with the file's own 260.5 rows/yr: "
      f"{s.mean() / s.std() * np.sqrt(T / years):.3f}")
for yr, g in df.groupby(df.index.year):
    print(f"calendar {yr}: strategy {((1+g['strategy']).prod()-1):+.2%}  "
          f"benchmark {((1+g['benchmark']).prod()-1):+.2%}")


# %% Cell 6: benchmark-relative statistics (beta, alpha, IR) with Newey-West errors
section("5. Benchmark-relative statistics")


def ols_nw(y, X, lags=5):
    """OLS with Newey-West (Bartlett) HAC standard errors. X must include a constant."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, u


X = np.column_stack([np.ones(T), bm.values])
b_, se_, resid = ols_nw(s.values, X)
alpha_ann = b_[0] * ANN
te = resid.std(ddof=2) * np.sqrt(ANN)
print(f"beta={b_[1]:.3f} (NW se {se_[1]:.3f})   corr={s.corr(bm):.3f}   R^2={s.corr(bm)**2:.3f}")
print(f"alpha={alpha_ann:.2%}/yr  NW t-stat={b_[0]/se_[0]:.2f}  "
      f"residual vol={te:.2%}  appraisal ratio={alpha_ann/te:.2f}")
active = s - bm
print(f"active return (strategy - benchmark)={active.mean()*ANN:.2%}/yr  "
      f"tracking error={active.std()*np.sqrt(ANN):.2%}  "
      f"information ratio={active.mean()/active.std()*np.sqrt(ANN):.2f}")
up, dn = bm > 0, bm < 0
print(f"up-capture={s[up].mean()/bm[up].mean():.2f}  down-capture={s[dn].mean()/bm[dn].mean():.2f}  "
      f"(strategy mean on benchmark-down days: {s[dn].mean():+.4%}, n={dn.sum()})")
# Henriksson-Merton: y = a + b*m + g*max(-m,0). g>0 means the book "knew" to be out on down days.
Xhm = np.column_stack([np.ones(T), bm.values, np.maximum(-bm.values, 0)])
bh, sh, _ = ols_nw(s.values, Xhm)
print(f"Henriksson-Merton timing gamma={bh[2]:.3f} (NW t={bh[2]/sh[2]:.2f}); "
      f"alpha left after timing term={bh[0]*ANN:.2%}/yr (t={bh[0]/sh[0]:.2f})")
print("cross-correlation corr(strategy_t, benchmark_{t+k}):",
      {k: round(s.corr(bm.shift(-k)), 3) for k in range(-3, 4)})


# %% Cell 7: sampling uncertainty of the Sharpe ratio
section("6. Sharpe ratio sampling uncertainty")
sr_d = s.mean() / s.std(ddof=1)
g3, g4 = stats.skew(s), stats.kurtosis(s, fisher=False)
se_d = np.sqrt((1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2) / (T - 1))   # Mertens (2002)
sr_a, se_a = sr_d * np.sqrt(ANN), se_d * np.sqrt(ANN)
print(f"annual Sharpe={sr_a:.2f}  analytic SE={se_a:.2f}  "
      f"95% CI=[{sr_a-1.96*se_a:.2f}, {sr_a+1.96*se_a:.2f}]")


def psr(sr_hat_d, sr_star_d):
    return stats.norm.cdf((sr_hat_d - sr_star_d) * np.sqrt(T - 1) /
                          np.sqrt(1 - g3 * sr_hat_d + (g4 - 1) / 4 * sr_hat_d ** 2))


sr_bm_d = bm.mean() / bm.std()
print(f"PSR(SR>0)={psr(sr_d, 0):.4f}   PSR(SR>benchmark {sr_bm_d*np.sqrt(ANN):.2f})="
      f"{psr(sr_d, sr_bm_d):.4f}   (single-test, before selection bias)")


def stationary_bootstrap_idx(n, mean_block, rng):
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    p = 1.0 / mean_block
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


rng = np.random.default_rng(SEED)
B, blk = 5000, 10
boot_sr, boot_alpha = np.empty(B), np.empty(B)
sv, bv = s.values, bm.values
for i in range(B):
    ix = stationary_bootstrap_idx(T, blk, rng)
    ss, bb = sv[ix], bv[ix]
    boot_sr[i] = ss.mean() / ss.std(ddof=1) * np.sqrt(ANN)
    cb = np.cov(ss, bb)
    beta_i = cb[0, 1] / cb[1, 1]
    boot_alpha[i] = (ss.mean() - beta_i * bb.mean()) * ANN
lo, hi = np.percentile(boot_sr, [2.5, 97.5])
print(f"stationary block bootstrap (mean block {blk}d, {B} reps): Sharpe 95% CI=[{lo:.2f}, {hi:.2f}]  "
      f"P(Sharpe<1)={np.mean(boot_sr < 1):.3f}")
alo, ahi = np.percentile(boot_alpha, [2.5, 97.5])
print(f"bootstrap alpha 95% CI=[{alo:.2%}, {ahi:.2%}]")


# %% Cell 8: multiple-testing adjustment for ~40 prompt variants
section("7. Selection bias from ~40 prompt variants")


def expected_max_sr(n, sd_ann):
    """Expected max of n iid N(0, sd^2) Sharpe estimates (Bailey & Lopez de Prado 2014)."""
    if n == 1:
        return 0.0
    z = (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * e))
    return sd_ann * z


noise_sd_ann = np.sqrt(ANN / (T - 1))   # dispersion of Sharpe estimates when no variant has skill
print(f"dispersion of annual Sharpe estimates under pure noise (T={T}): {noise_sd_ann:.2f}")
rows = []
for n_eff in [1, 5, 10, 20, 40]:
    for sd_ann in [0.5, noise_sd_ann, 1.0]:
        sr0 = expected_max_sr(n_eff, sd_ann)
        rows.append({"N_eff": n_eff, "cross_trial_SR_sd": round(sd_ann, 2),
                     "hurdle_SR0": round(sr0, 2),
                     "DSR": round(psr(sr_d, sr0 / np.sqrt(ANN)), 3)})
dsr = pd.DataFrame(rows)
print(dsr.to_string(index=False))

t_sr = sr_a * np.sqrt(years)
p1 = 2 * (1 - stats.norm.cdf(t_sr))
print(f"\nSharpe t-stat ~ SR*sqrt(years) = {t_sr:.2f}, two-sided p={p1:.4f}")
for n in [10, 20, 40]:
    p_adj = min(1.0, n * p1)
    t_adj = stats.norm.ppf(1 - p_adj / 2)
    print(f"Bonferroni haircut N={n}: adj p={p_adj:.3f}  haircut Sharpe={t_adj/np.sqrt(years):.2f} "
          f"({1 - t_adj/t_sr:.0%} haircut)")
t_alpha = b_[0] / se_[0]
p_alpha = 2 * (1 - stats.norm.cdf(abs(t_alpha)))
print(f"alpha NW t={t_alpha:.2f}, p={p_alpha:.4f}, Bonferroni x40 p={min(1, 40*p_alpha):.3f}")


# %% Cell 9: sub-period stability and the GPT-4o training-cutoff split
section("8. Sub-period stability")


def sub(g):
    return pd.Series({
        "days": len(g),
        "strat_ret": (1 + g["strategy"]).prod() - 1,
        "bench_ret": (1 + g["benchmark"]).prod() - 1,
        "strat_SR": g["strategy"].mean() / g["strategy"].std() * np.sqrt(ANN),
        "bench_SR": g["benchmark"].mean() / g["benchmark"].std() * np.sqrt(ANN),
    })


print("by year:")
print(df.groupby(df.index.year).apply(sub).round(3))
print("\nby half-year:")
print(df.groupby([df.index.year, (df.index.month - 1) // 6 + 1]).apply(sub).round(3))
pre, post = df.loc[:pd.Timestamp(GPT4O_CUTOFF) - pd.Timedelta(days=1)], df.loc[GPT4O_CUTOFF:]
print(f"\nsplit at assumed GPT-4o training cutoff {GPT4O_CUTOFF}:")
print(pd.DataFrame({"inside_training_window": sub(pre), "after_cutoff": sub(post)}).round(3))
monthly = (1 + df).resample("ME").prod() - 1
print(f"\nmonths with strategy > 0: {(monthly['strategy'] > 0).sum()} of {len(monthly)};  "
      f"months strategy beat benchmark: {(monthly['strategy'] > monthly['benchmark']).sum()}")
roll = s.rolling(126).apply(lambda x: x.mean() / x.std() * np.sqrt(ANN))
print(f"rolling 6-month Sharpe: min={roll.min():.2f}  median={roll.median():.2f}  max={roll.max():.2f}")


# %% Cell 10: re-measure after removing the 19 non-trading-day rows
section("9. Face-value stats on NYSE sessions only (holiday rows removed)")
df_tr = df.drop(present.index)
s_tr, b_tr = df_tr["strategy"], df_tr["benchmark"]
print(f"rows={len(df_tr)}  strategy Sharpe={s_tr.mean()/s_tr.std()*np.sqrt(ANN):.2f}  "
      f"benchmark Sharpe={b_tr.mean()/b_tr.std()*np.sqrt(ANN):.2f}  "
      f"strategy total return={(1+s_tr).prod()-1:+.1%}  benchmark={(1+b_tr).prod()-1:+.1%}")


# %% Cell 11: transaction-cost sensitivity (turnover is unknown, so this is a grid)
section("10. Transaction-cost sensitivity")
mu_a, vol_a = s.mean() * ANN, s.std() * np.sqrt(ANN)
grid = []
for turnover in [0.05, 0.10, 0.25, 0.50, 1.00]:          # fraction of book traded per day
    row = {"daily_turnover": f"{turnover:.0%}"}
    for bps in [2, 5, 10, 20]:                             # all-in cost per unit traded
        drag = turnover * bps / 1e4 * ANN
        row[f"SR@{bps}bps"] = round((mu_a - drag) / vol_a, 2)
    row["breakeven_bps_to_SR1"] = round((mu_a - 1.0 * vol_a) / (turnover * ANN) * 1e4, 1)
    row["breakeven_bps_to_bench_SR"] = round(
        (mu_a - (bm.mean() / bm.std() * np.sqrt(ANN)) * vol_a) / (turnover * ANN) * 1e4, 1)
    grid.append(row)
print(pd.DataFrame(grid).to_string(index=False))
print("(cost = commission + half-spread + impact per unit of notional traded; Sharpe with rf=0)")
