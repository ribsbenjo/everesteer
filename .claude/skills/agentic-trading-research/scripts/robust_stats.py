#!/usr/bin/env python3
"""Dependence-aware statistics for trading / forecasting evaluations (numpy + pandas + stdlib only).

Methods used across the papers:
  * moving-block bootstrap CI of a mean (SAE; Na et al. used 10-day blocks, 10,000 draws)
  * paired block bootstrap of the difference between two aligned return series
  * Wilcoxon signed-rank (normal approximation with tie correction) on aligned returns (SAE)
  * two-proportion z-test for rates such as AttackSuccess / ASR (SAE, Na et al.)
  * rule-of-three upper bound for zero-event arms (Fonseca)
  * attack success rate on attackable days (Na et al.)
  * max drawdown, CVaR (expected shortfall), Sharpe
  * probabilistic and deflated Sharpe ratio (Bailey & Lopez de Prado 2014, cited by the survey;
    the formula itself is not reproduced in the PDFs)

CLI
  python robust_stats.py summary  --csv returns.csv --col strat [--periods 252] [--block 10]
  python robust_stats.py paired-ci --csv returns.csv --a strat --b baseline [--block 10]
  python robust_stats.py dsr --sr 0.06 --n 750 --skew -0.3 --kurt 5 --trials 25 --var-trials 0.0004
  python robust_stats.py twoprop --x1 183 --n1 183 --x2 35 --n2 183
CSV: first column is the date index; other columns are per-period returns.
"""

from __future__ import annotations

import argparse
import math
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
def max_drawdown(returns, compounded: bool = True) -> float:
    """Most negative (E_t - max_{s<=t} E_s) / max_{s<=t} E_s; returned as a negative fraction."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    eq = np.cumprod(1.0 + r) if compounded else 1.0 + np.cumsum(r)
    eq = np.concatenate([[1.0], eq])
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak))


def cvar(returns, alpha: float = 0.95) -> float:
    """Expected shortfall: mean of returns at or below the (1 - alpha) quantile (a negative number)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    q = np.quantile(r, 1.0 - alpha)
    return float(r[r <= q].mean())


def sharpe(returns, periods_per_year: float | None = None) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    s = r.std(ddof=1)
    if s == 0 or r.size < 2:
        return float("nan")
    sr = r.mean() / s
    return float(sr * math.sqrt(periods_per_year)) if periods_per_year else float(sr)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _default_block(n: int) -> int:
    return max(2, int(round(n ** (1.0 / 3.0))))


def block_bootstrap(x, stat=np.mean, block: int | None = None, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap distribution of stat(x). Keeps short-range autocorrelation."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    b = block or _default_block(n)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    samples = x[idx.reshape(n_boot, -1)[:, :n]]
    return np.apply_along_axis(stat, 1, samples) if stat is not np.mean else samples.mean(axis=1)


def block_bootstrap_ci(x, stat=np.mean, block: int | None = None, n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    d = block_bootstrap(x, stat, block, n_boot, seed)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def paired_block_bootstrap_ci(a, b, block: int | None = None, n_boot: int = 5000, alpha: float = 0.05,
                              seed: int = 0) -> dict:
    """CI of mean(a - b) on aligned periods (a, b: pd.Series or arrays of equal length)."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, join="inner")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    dist = block_bootstrap(diff, np.mean, block, n_boot, seed)
    lo, hi = np.quantile(dist, alpha / 2), np.quantile(dist, 1 - alpha / 2)
    return dict(mean_diff=float(diff.mean()), ci=(float(lo), float(hi)), n=int(diff.size),
                share_boot_le_0=float((dist <= 0).mean()), clears_zero=bool(lo > 0 or hi < 0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank(a, b=None) -> dict:
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie correction (zero diffs dropped).

    Note: like any per-observation test it assumes independent pairs; with autocorrelated returns
    prefer the paired block bootstrap and treat this as a secondary check.
    """
    d = np.asarray(a, dtype=float) - (0.0 if b is None else np.asarray(b, dtype=float))
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = d.size
    if n < 10:
        return dict(n=n, z=float("nan"), p=float("nan"), note="too few non-zero pairs for normal approximation")
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = (w_plus - mean) / math.sqrt(var)
    return dict(n=n, w_plus=float(w_plus), z=float(z), p=float(2 * (1 - _N.cdf(abs(z)))))


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided pooled two-proportion z-test. Cross-check small cells with Fisher's exact test."""
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    return dict(p1=p1, p2=p2, z=z, p=float(2 * (1 - _N.cdf(abs(z)))) if se > 0 else float("nan"))


def rule_of_three(n: int, conf: float = 0.95) -> dict:
    """Upper confidence bound on a rate when 0 events were observed in n trials."""
    return dict(approx=3.0 / n, exact=1.0 - (1.0 - conf) ** (1.0 / n))


def attack_success_rate(clean, attacked, target) -> dict:
    """ASR on attackable days only (Na et al.): #{clean != target and attacked == target} / #{clean != target}."""
    clean, attacked = np.asarray(clean), np.asarray(attacked)
    target = np.broadcast_to(np.asarray(target), clean.shape)
    attackable = clean != target
    n = int(attackable.sum())
    succ = int((attackable & (attacked == target)).sum())
    return dict(asr=succ / n if n else float("nan"), successes=succ, attackable=n)


# ---------------------------------------------------------------------------
# Sharpe deflation
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > sr_benchmark). sr is per-period (not annualised); kurt is raw (normal = 3)."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or n < 2:
        return float("nan")
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum per-period SR among n_trials unskilled strategies with SR variance var_trials."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(var_trials) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1.0 / n_trials)
                                    + EULER_GAMMA * _N.inv_cdf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_trials: float) -> dict:
    """DSR = PSR against the SR you would expect from the best of n_trials pure-noise strategies."""
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return dict(sr=sr, sr0=sr0, dsr=probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0),
                psr_vs_zero=probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0))


def summary(returns: pd.Series, periods_per_year: float | None = None, block: int | None = None, seed: int = 0) -> dict:
    r = pd.Series(returns).dropna().astype(float)
    lo, hi = block_bootstrap_ci(r.to_numpy(), block=block, seed=seed)
    return dict(n=int(r.size), mean=float(r.mean()), vol=float(r.std(ddof=1)),
                sharpe=sharpe(r, periods_per_year), max_drawdown=max_drawdown(r),
                cvar95=cvar(r, 0.95), cvar99=cvar(r, 0.99), mean_ci95=(lo, hi),
                skew=float(r.skew()), kurt=float(r.kurt() + 3.0), block=block or _default_block(int(r.size)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        pass
    return df.sort_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("--csv", required=True); s.add_argument("--col", required=True)
    s.add_argument("--periods", type=float, default=None); s.add_argument("--block", type=int, default=None)
    p = sub.add_parser("paired-ci"); p.add_argument("--csv", required=True); p.add_argument("--a", required=True)
    p.add_argument("--b", required=True); p.add_argument("--block", type=int, default=None)
    d = sub.add_parser("dsr"); d.add_argument("--sr", type=float, required=True, help="per-period Sharpe of the chosen strategy")
    d.add_argument("--n", type=int, required=True); d.add_argument("--skew", type=float, default=0.0)
    d.add_argument("--kurt", type=float, default=3.0, help="raw kurtosis (normal = 3)")
    d.add_argument("--trials", type=int, required=True, help="number of variants tried")
    d.add_argument("--var-trials", type=float, required=True, help="variance of per-period Sharpe across the trials")
    t = sub.add_parser("twoprop")
    for k in ("x1", "n1", "x2", "n2"):
        t.add_argument(f"--{k}", type=int, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "summary":
        res = summary(_read(a.csv)[a.col], a.periods, a.block)
    elif a.cmd == "paired-ci":
        df = _read(a.csv)
        res = paired_block_bootstrap_ci(df[a.a], df[a.b], block=a.block)
        res["wilcoxon"] = wilcoxon_signed_rank(df[a.a].to_numpy(), df[a.b].to_numpy())
    elif a.cmd == "dsr":
        res = deflated_sharpe_ratio(a.sr, a.n, a.skew, a.kurt, a.trials, a.var_trials)
    else:
        res = two_proportion_z(a.x1, a.n1, a.x2, a.n2)
    for k, v in res.items():
        print(f"{k:>16}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
