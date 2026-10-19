"""
round.py: one command per live round. Run it in the VS Code terminal (or F5 -> "Round").

    python round.py             1) asks for our competition points for the last submitted round(s)
                                2) backtests the current main.py (skips if nothing changed)
                                3) marks this version as submitted for the next round
    python round.py --status    live-round breakdown + backtest calibration, no questions
    python round.py --recheck   re-run every submitted version with the CURRENT tester and check whether
                                it explains our real points better than the tester we had at the time

Extra flags go to the backtest: --stress, --rounds 60, --inprocess.
Only the tester PC runs this file: it owns scorekeeper_log.csv and runs/.
"""

import argparse
import hashlib
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import calibrate
import stats

CALIBRATE_ON = "points"  # FILL: "points" (leaderboard points) or "score" (raw metric, if the platform shows it)
# Points are on a different scale than the backtest metric, so only the ratio k (metric -> points) makes sense.
CALIB_MODE = "ratio" if CALIBRATE_ON == "points" else "auto"
MAIN = "main.py"


# ---------------------------------------------------------------------------
# prompts (Enter = skip; Ctrl+Z / closed input also counts as skip)
# ---------------------------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    try:
        v = input(prompt).strip()
    except EOFError:
        v = ""
    return v or default


def ask_float(prompt: str):
    while True:
        v = ask(prompt)
        if not v:
            return None
        try:
            return float(v.replace(",", "."))
        except ValueError:
            print("  a number please, or Enter to skip")


# ---------------------------------------------------------------------------
# log helpers
# ---------------------------------------------------------------------------
def load_log() -> pd.DataFrame:
    return calibrate.load_log() if calibrate.LOG.exists() else pd.DataFrame(columns=["row_type"])


def submits(log: pd.DataFrame) -> pd.DataFrame:
    s = log[log["row_type"] == "submit"].copy()
    if not s.empty:
        s["round_id"] = s["round_id"].astype(int)
    return s.drop_duplicates("round_id", keep="last")


def pending(log: pd.DataFrame) -> pd.DataFrame:
    scored = set(calibrate.round_rows(log)["round_id"]) if (log["row_type"] == "round").any() else set()
    s = submits(log)
    return s[~s["round_id"].isin(scored)].sort_values("round_id") if not s.empty else s


def append(row: dict) -> None:
    log = load_log()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(calibrate.LOG, index=False)


def file_hash(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def enter_points() -> bool:
    """Ask for points of every submitted-but-unscored round. Returns True if anything was entered."""
    entered = False
    for _, s in pending(load_log()).iterrows():
        rid = int(s["round_id"])
        print(stats.bold(f"\nRound {rid} was submitted with '{s['label']}' (run {s['run_id']})."))
        pts = ask_float(f"  Competition points for round {rid}? [Enter = not scored yet]: ")
        if pts is None:
            break  # later rounds can't be scored before this one
        score = ask_float("  Platform score / metric value, if shown [Enter = skip]: ")
        rank = ask_float("  Leaderboard rank [Enter = skip]: ")
        replay = ask_float("  Replay score from backtest.py --replay [Enter = skip]: ") if score is not None else None
        actual = pts
        if CALIBRATE_ON == "score":
            if score is None:
                print(stats.warn("  No platform score: calibrating this round on points instead."))
            else:
                actual = score
        print()
        calibrate.record_round(rid, actual, replay=replay, run_id=s["run_id"], mode=CALIB_MODE, verbose=False,
                               extra={"points": pts, "platform_score": score, "rank": rank})
        entered = True
    return entered


def show_status() -> None:
    log = load_log()
    if not (log["row_type"] == "round").any():
        print("No live rounds scored yet.")
    else:
        res = calibrate.compute(calibrate.round_rows(log), CALIB_MODE)
        print()
        print(stats.live_breakdown(res))
        print()
        print(stats.bold("=== Backtest calibration ==="))
        calibrate.report(res)
    p = pending(log)
    if not p.empty:
        print(stats.warn(f"\nWaiting for points: round(s) {', '.join(map(str, p['round_id']))}"))


def run_or_reuse(args) -> dict | None:
    log = load_log()
    bts = log[log["row_type"] == "backtest"]
    last = bts.iloc[-1] if not bts.empty else None
    h = file_hash(MAIN)
    if (last is not None and str(last["main_hash"]) == h
            and str(last["backtest_version"]) == backtest.TESTER_VERSION and not args.stress):
        again = ask(f"\nmain.py and tester unchanged since run {last['run_id']} ('{last['label']}'). "
                    "Re-run backtest? [y/N]: ").lower()
        if again != "y":
            return last.to_dict()
    default = last["label"] if last is not None and str(last["main_hash"]) == h else ""
    label = ""
    while not label or label.lower() in {"y", "n", "yes", "no"}:
        if label:
            print("  That looks like a y/n answer. Describe the change, e.g. 'added reversal 0.5'.")
        label = ask(f"\nWhat changed in main.py / the tester? (label{f', Enter = {default!r}' if default else ''}): ",
                    default)
    out = backtest.run_backtest(backtest.make_args(
        main=MAIN, label=label, stress=args.stress, rounds=args.rounds, inprocess=args.inprocess))
    return out["row"]


def mark_submitted(row: dict) -> None:
    log = load_log()
    s = submits(log)
    nxt = int(s["round_id"].max()) + 1 if not s.empty else 1
    if row.get("leak_test") == "fail" or str(row.get("validation", "pass")) != "pass":
        print(stats.bad("\nThis version FAILED the leak test or validation. Submitting it is risky."))
    ans = ask(f"\nSubmit '{row['label']}' as round {nxt}? [y = yes, number = other round, Enter = no]: ").lower()
    if ans.isdigit():
        nxt, ans = int(ans), "y"
    if ans != "y":
        print("Not marked as submitted.")
        return
    append({
        "row_type": "submit", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "round_id": nxt, "run_id": row["run_id"], "label": row["label"], "main_hash": row["main_hash"],
        "backtest_version": row["backtest_version"], "snapshot": row.get("snapshot"),
    })
    print(stats.ok(f"Round {nxt} <- '{row['label']}' (snapshot {row.get('snapshot')})."))
    print("Submit it on the platform now. Run round.py again once the round is scored to enter its points.")


def recheck(args) -> None:
    """Did the tester upgrade make the backtest explain our past real points better?"""
    log = load_log()
    rounds = calibrate.round_rows(log)
    if len(rounds) < 2:
        print("Need at least 2 scored rounds to recheck.")
        return
    snaps = log[log["row_type"] == "backtest"].set_index("run_id")["snapshot"]
    cache, rows = {}, []
    print(f"Re-running {rounds['main_hash'].nunique()} submitted version(s) with tester {backtest.TESTER_VERSION} ...")
    for _, r in rounds.iterrows():
        snap = snaps.get(str(r["run_id"]))
        path = Path(str(snap)) / "main.py" if isinstance(snap, str) else None
        if path is None or not path.exists():
            print(stats.warn(f"  round {r['round_id']}: no snapshot of main.py, skipped"))
            continue
        key = str(r["main_hash"])
        if key not in cache:
            out = backtest.run_backtest(backtest.make_args(main=str(path), label=f"recheck R{r['round_id']}",
                                                           rounds=args.rounds, inprocess=True),
                                        log=False, verbose=False)
            cache[key] = out["row"]["bt_mean"]
        pts = r["points"] if "points" in r and pd.notna(r.get("points")) else r["actual_score"]
        rows.append({"round": r["round_id"], "version": str(r["label"])[:28], "old_tester": r["backtest_version"],
                     "old_bt": r["bt_mean"], "new_bt": cache[key], "points": pts})
    t = pd.DataFrame(rows)
    if len(t) < 2:
        print("Not enough rounds with snapshots.")
        return
    print(t.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    def fit_error(bt: pd.Series) -> float:
        """RMSE of points after the best straight-line mapping from backtest to points."""
        y = t["points"].astype(float)
        if len(t) >= 3 and bt.nunique() >= 2:
            a, b = np.polyfit(bt.astype(float), y, 1)
            return float(np.sqrt(np.mean((y - (a * bt + b)) ** 2)))
        return float(np.sqrt(np.mean((y - y.mean()) ** 2)))

    e_old, e_new = fit_error(t["old_bt"]), fit_error(t["new_bt"])
    print(f"\nFit error vs real points: old tester {e_old:.4f} | current tester {e_new:.4f}")
    if t["new_bt"].nunique() < 3:
        print("Fewer than 3 different submitted versions: any tester fits them equally well. "
              "Recheck again after more version changes.")
    elif e_new < 0.95 * e_old:
        print(stats.ok("Current tester explains past rounds better. Keep the upgrade."))
    elif e_new > 1.05 * e_old:
        print(stats.bad("Current tester explains past rounds WORSE. Reconsider the last tester change."))
    else:
        print("About the same. The tester change neither helped nor hurt on past rounds.")
    print(stats.warn("Few rounds = noisy. Don't keep a tester change because of 1-2 rounds."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--rounds", type=int, default=60)
    args = ap.parse_args()

    if args.status:
        show_status()
        return
    if args.recheck:
        recheck(args)
        return

    print(stats.bold(f"=== Everesteer round tool | tester {backtest.TESTER_VERSION} | "
                     f"{datetime.now():%H:%M:%S} ==="))
    if enter_points():
        show_status()
    row = run_or_reuse(args)
    if row is not None:
        mark_submitted(row)


if __name__ == "__main__":
    main()
