"""
calibrate.py: how well did the backtest predict the real round? (Scorekeeper KPI)

Normally round.py calls this for you after asking for our points. Manual use:
    python calibrate.py --round 1 --actual 0.031                       # round 1 = benchmark
    python calibrate.py --round 2 --actual 0.012 --replay 0.012        # replay from backtest.py --replay
    python calibrate.py --round 3 --actual 0.020 --run 192233          # link a specific backtest run
    python calibrate.py --show                                         # reprint the table

Each round links to a backtest row in scorekeeper_log.csv (default: the latest one, so run
backtest.py on the submitted version before submitting). Rows are only ever appended;
if a round is entered twice, the latest entry wins.

    k_n    = sum(actual_1..n) / sum(bt_1..n)               ratio mode
    pred_n = k_(n-1) * bt_n
    z_n    = (actual_n - pred_n) / (|k_(n-1)| * bt_std_n)
Additive mode (scores near zero, e.g. correlation): d = mean(actual - bt), pred_n = bt_n + d_(n-1),
z_n = (actual_n - pred_n) / bt_std_n.
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import stats
from backtest import CONFIG

LOG = Path(CONFIG["log_path"])
REPLAY_TOL = 1e-4   # FILL: how close replay must be to the platform score (depends on metric rounding)
# auto mode: use the ratio k only if the backtest mean is clearly away from zero
# (|bt_mean| > 2 standard errors); correlation-type scores usually end up additive.


def load_log() -> pd.DataFrame:
    if not LOG.exists():
        raise SystemExit(f"{LOG} not found: run backtest.py first.")
    log = pd.read_csv(LOG, dtype={"run_id": str, "backtest_version": str, "main_hash": str})
    if "row_type" not in log:
        log["row_type"] = "backtest"
    return log


def round_rows(log: pd.DataFrame) -> pd.DataFrame:
    r = log[log["row_type"] == "round"].copy()
    if r.empty:
        return r
    r["round_id"] = r["round_id"].astype(int)
    return r.drop_duplicates("round_id", keep="last").sort_values("round_id").reset_index(drop=True)


def compute(rounds: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Recompute k, pred, z for every round from scratch (so re-entries stay consistent)."""
    out = rounds.copy()
    for col in ["k", "pred", "z"]:
        out[col] = np.nan
    out["mode"] = ""
    for i in range(len(out)):
        bt, sd, act = out.loc[i, ["bt_mean", "bt_std", "actual_score"]].astype(float)
        prev = out.iloc[:i]
        sum_bt_all = prev["bt_mean"].sum() + bt
        m = mode
        if m == "auto":
            n_bt = max(float(out.loc[i, "n_rounds"]), 1.0) if "n_rounds" in out else 1.0
            m = "ratio" if abs(sum_bt_all / (i + 1)) > 2 * sd / np.sqrt(n_bt) else "additive"
        out.loc[i, "mode"] = m
        if m == "ratio":
            out.loc[i, "k"] = (prev["actual_score"].sum() + act) / sum_bt_all if sum_bt_all else np.nan
            if i > 0:
                s_bt = prev["bt_mean"].sum()
                k_prev = prev["actual_score"].sum() / s_bt if s_bt else np.nan
                out.loc[i, "pred"] = k_prev * bt
                out.loc[i, "z"] = (act - k_prev * bt) / (abs(k_prev) * sd) if sd > 0 and k_prev else np.nan
        else:
            out.loc[i, "k"] = ((prev["actual_score"] - prev["bt_mean"]).sum() + act - bt) / (i + 1)
            if i > 0:
                d_prev = (prev["actual_score"] - prev["bt_mean"]).mean()
                out.loc[i, "pred"] = bt + d_prev
                out.loc[i, "z"] = (act - bt - d_prev) / sd if sd > 0 else np.nan
    return out


def report(res: pd.DataFrame) -> None:
    cols = ["round_id", "label", "backtest_version", "bt_mean", "bt_std", "actual_score",
            "replay_score", "pred", "z", "k", "mode"]
    cols = [c for c in cols if c in res]
    pd.set_option("display.width", 160)
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    z = res["z"].dropna()
    k = res["k"].iloc[-1]
    mode = res["mode"].iloc[-1]
    if z.empty:
        print("\nBenchmark set (round 1). k is provisional: one round is mostly luck.")
    else:
        fid = (z.abs() <= 1).mean()
        print(f"\nFidelity {fid:.0%} of {len(z)} rounds with |z|<=1 (ideal ~68%) | mean z {z.mean(): .2f} | "
              f"{'k' if mode == 'ratio' else 'offset d'} {k: .3f}")
        print("Verdict:", verdict(fid, z.mean(), k, mode, len(z)))

    alerts = []
    if "replay_score" in res:
        rp = res.dropna(subset=["replay_score"])
        # replay is in metric units: compare with the platform's metric score when we have it
        ref = rp["platform_score"].fillna(rp["actual_score"]) if "platform_score" in rp else rp["actual_score"]
        for (_, r), real in zip(rp.iterrows(), ref):
            if abs(r.replay_score - real) > REPLAY_TOL:
                alerts.append(f"round {r.round_id}: replay {r.replay_score:.4f} != platform {real:.4f} "
                              "-> scoring/data/alignment BUG, fix backtest now")
    resid = (res["actual_score"] - res["pred"]).dropna()
    if len(resid) >= 3 and (np.sign(resid.tail(3)).nunique() == 1):
        side = "optimistic" if resid.iloc[-1] < 0 else "pessimistic"
        alerts.append(f"same-direction miss 3 rounds in a row: backtest looks {side}")
    if (z.abs() > 2).sum() >= 2:
        alerts.append(f"|z| > 2 on {(z.abs() > 2).sum()} rounds: noise estimate or backtest is off")
    if mode == "ratio" and k < 0:
        alerts.append("k is negative: sign / target / date-alignment problem")
    d_bt, d_act = res["bt_mean"].diff(), res["actual_score"].diff()
    changed = d_bt.abs() > 1e-12  # only rounds where the submitted version changed
    if changed.sum() >= 5:
        agree = (np.sign(d_bt[changed]) == np.sign(d_act[changed])).mean()
        print(f"Change direction agreement over {changed.sum()} version changes: {agree:.0%} (chance = 50%)")
    for a in alerts:
        print(stats.bad("ALERT: " + a))


def verdict(fid, mz, k, mode, n) -> str:
    if n < 5:
        return "too few rounds to judge; keep logging."
    if mode == "ratio" and k < 0:
        return "backtest points the wrong way; check target and alignment."
    if fid < 0.4:
        return f"backtest is {'optimistic' if mz < 0 else 'pessimistic'} or overconfident; see the adjust table."
    if fid > 0.95:
        return "noise band too wide (bt_std too large); results are uninformative."
    if mode == "ratio" and k < 0.5:
        return "roughly calibrated in noise, but ~2x optimistic: add costs / lag, check holdout."
    return "backtest tracks live rounds reasonably; trust it for keep/drop decisions."


def record_round(round_id: int, actual: float, replay=None, run_id=None, extra=None, mode="auto", verbose=True):
    """Append a scored live round linked to a backtest run (default: the latest), print the KPI table."""
    log = load_log()
    bts = log[log["row_type"] == "backtest"]
    if run_id is not None:
        bts = bts[bts["run_id"].astype(str) == str(run_id)]
    if bts.empty:
        raise SystemExit("No matching backtest row in the log.")
    bt = bts.iloc[-1]
    row = {
        "row_type": "round", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "round_id": round_id, "run_id": bt["run_id"], "label": bt["label"],
        "main_hash": bt["main_hash"], "backtest_version": bt["backtest_version"],
        "bt_mean": bt["bt_mean"], "bt_std": bt["bt_std"], "n_rounds": bt["n_rounds"],
        "actual_score": actual, "replay_score": replay, **(extra or {}),
    }
    row = {k: v for k, v in row.items() if v is not None}
    print(f"Round {round_id} linked to backtest run {bt['run_id']} ({bt['label']}, hash {bt['main_hash']})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)

    res = compute(round_rows(log), mode)
    if verbose:
        report(res)
    latest = res[res["round_id"] == round_id].iloc[0]
    for col in ["k", "pred", "z", "mode"]:
        log.loc[log.index[-1], col] = latest[col]
    log.to_csv(LOG, index=False)
    return res


def show(mode="auto"):
    rounds = round_rows(load_log())
    if rounds.empty:
        print("No rounds entered yet.")
        return rounds
    res = compute(rounds, mode)
    report(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--actual", type=float)
    ap.add_argument("--replay", type=float, default=None)
    ap.add_argument("--run", default=None, help="run_id of the backtest row for the submitted version")
    ap.add_argument("--mode", choices=["auto", "ratio", "additive"], default="auto")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show(args.mode)
    elif args.round is None or args.actual is None:
        ap.error("--round and --actual are required (or use --show)")
    else:
        record_round(args.round, args.actual, args.replay, args.run, mode=args.mode)


if __name__ == "__main__":
    main()
