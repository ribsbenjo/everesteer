"""
stats.py: small text breakdowns for the backtest and the live rounds (VS Code terminal friendly).
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(errors="replace")  # never crash on a character the terminal can't show
except AttributeError:
    pass
_UNICODE = (sys.stdout.encoding or "").lower().replace("-", "").startswith("utf")
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if os.name == "nt" and _COLOR:
    os.system("")  # turns on ANSI colours in older Windows consoles


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else str(text)


def ok(t):
    return _c(t, "32")


def bad(t):
    return _c(t, "31")


def warn(t):
    return _c(t, "33")


def bold(t):
    return _c(t, "1")


def spark(values) -> str:
    v = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(v) == 0:
        return ""
    chars = "▁▂▃▄▅▆▇█" if _UNICODE else "_.-~=+*#"
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return chars[len(chars) // 2] * len(v)
    idx = np.round((v - lo) / (hi - lo) * (len(chars) - 1)).astype(int)
    return "".join(chars[i] for i in idx)


def neutral_score(metric: str) -> float:
    """Score of a no-information submission."""
    return {"hit_rate": 0.5, "log_loss": float(np.log(2))}.get(metric, 0.0)


def longest_streak(mask) -> int:
    run = best = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def max_drawdown(s: pd.Series) -> float:
    cum = s.cumsum()
    return float((cum - cum.cummax()).min())


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def backtest_breakdown(res: pd.DataFrame, metric: str, higher_is_better: bool) -> str:
    """res: per-round scores indexed by date with columns team, momentum, random, status."""
    sign = 1 if higher_is_better else -1
    team = res["team"].astype(float)
    edge = sign * (team - neutral_score(metric))  # > 0 = better than a no-information submission
    good = edge > 0
    n = len(team)
    best_i = (sign * team).idxmax()
    worst_i = (sign * team).idxmin()
    beat_mom = (sign * (team - res["momentum"]) > 0).mean()
    beat_rnd = (sign * (team - res["random"]) > 0).mean()
    roll = team.rolling(min(10, n)).mean().dropna()

    span = (res.index[-1] - res.index[0]).days if n > 1 else 0
    freq = "M" if span > 90 else "W" if span > 14 else "D"
    by = team.groupby(res.index.to_period(freq)).agg(["mean", "size"]).tail(6)
    name = (lambda p: f"wk {p.start_time:%m-%d}") if freq == "W" else str
    by_txt = " | ".join(f"{name(p)}: {m:+.4f} ({int(k)})" for p, (m, k) in by.iterrows())

    lines = [
        bold("--- Breakdown ---"),
        f"Good rounds   {good.sum()}/{n} ({good.mean():.0%})   beat momentum {beat_mom:.0%}   beat random {beat_rnd:.0%}",
        f"Best  {team[best_i]: .4f} ({best_i.date()})   Worst {team[worst_i]: .4f} ({worst_i.date()})   "
        f"longest bad streak {longest_streak(~good)}",
        f"Cumulative edge {edge.sum(): .4f}   max drawdown {max_drawdown(edge): .4f}   "
        f"corr with momentum {_corr(team, res['momentum']): .2f}",
        f"Rolling 10-round mean: first {roll.iloc[0]: .4f}  last {roll.iloc[-1]: .4f}" if len(roll) else "",
        f"Last {min(30, n)} rounds  {spark(team.tail(30))}",
        f"By period    {by_txt}",
    ]
    if "status" in res and (res["status"] != "ok").any():
        lines.append(bad(f"Missed/invalid {(res['status'] != 'ok').sum()} rounds"))
    return "\n".join(l for l in lines if l)


def live_breakdown(rounds: pd.DataFrame) -> str:
    """rounds: one row per scored live round (calibrate.round_rows + compute)."""
    if rounds.empty:
        return "No live rounds scored yet."
    r = rounds.copy()
    pts = r["points"].astype(float) if "points" in r else r["actual_score"].astype(float)
    r["pts"] = pts
    r["cum"] = pts.cumsum()
    r["vs_prev"] = pts.diff()
    cols = ["round_id", "label", "bt_mean", "pts", "cum", "vs_prev"]
    if "rank" in r and r["rank"].notna().any():
        r["rank"] = r["rank"].astype(float).round().astype("Int64")
        cols.append("rank")
    if "platform_score" in r and r["platform_score"].notna().any():
        cols.append("platform_score")
    table = r[cols].rename(columns={"round_id": "round", "label": "version", "pts": "points"})
    table["version"] = table["version"].astype(str).str.slice(0, 28)

    best = r.loc[pts.idxmax()]
    worst = r.loc[pts.idxmin()]
    lines = [
        bold("=== Live rounds ==="),
        table.to_string(index=False, float_format=lambda x: f"{x: .4f}" if abs(x) < 1 else f"{x: .2f}"),
        "",
        f"Total {pts.sum():.2f} pts over {len(r)} rounds | avg {pts.mean():.2f} | "
        f"last 3 avg {pts.tail(3).mean():.2f} | best R{int(best.round_id)} ({best.pts:.2f}) | "
        f"worst R{int(worst.round_id)} ({worst.pts:.2f})",
        f"Points  {spark(pts)}",
    ]
    if "rank" in r and r["rank"].notna().sum() >= 2:
        rk = r["rank"].dropna()
        trend = ok("better") if rk.iloc[-1] < rk.iloc[0] else bad("worse") if rk.iloc[-1] > rk.iloc[0] else "flat"
        lines.append(f"Rank    {int(rk.iloc[0])} -> {int(rk.iloc[-1])} ({trend})")
    by_ver = r.groupby("label", sort=False)["pts"].agg(["size", "mean"])
    lines.append("By version  " + " | ".join(f"{v[:24]}: {m:.2f} avg ({int(k)})" for v, (k, m) in by_ver.iterrows()))
    if len(r) >= 4 and r["bt_mean"].nunique() >= 3:
        lines.append(f"Backtest vs points corr {_corr(r['bt_mean'].astype(float), pts): .2f} "
                     f"over {len(r)} rounds (noisy until ~6+ rounds)")
    return "\n".join(lines)
