# Everesteer Night: backtest skeleton

| File | Owner | What it does |
|---|---|---|
| `main.py` | Builder | Our forecast. `signal(prices)` → date × asset panel, `forecast(prices)` → one value per asset. Baseline = 20-bar momentum. |
| `round.py` | Scorekeeper | **The one command per round.** Asks for our competition points, backtests the current `main.py`, marks it submitted. |
| `backtest.py` | Scorekeeper | Walk-forward replay of `main.py`: fresh process per round, validation, deadline, leak test, metric, stats, log, snapshot. |
| `calibrate.py` | — | Backtest prediction vs real points per round (k, z, fidelity, alerts). Called by `round.py`. |
| `stats.py` | — | The text breakdowns (good-round rate, streaks, drawdown, sparklines, points by version). |
| `make_fake_data.py` | — | Synthetic `data/prices.csv` so everything runs before 18:00. Ignore once real data arrives. |

Generated while running: `scorekeeper_log.csv` (every backtest, submission and scored round) and `runs/<run_id>_<label>/` (a copy of `main.py` + `backtest.py` for every backtest run, so any version can be restored).

Only pandas + numpy (scipy is not installed).

## VS Code

Open the folder, then **Run and Debug (Ctrl+Shift+D)**, pick a configuration, F5:

| Configuration | Use it |
|---|---|
| **Round: points -> backtest -> submit** | Once per live round. |
| Round (stress test) | Same, with random missing assets/rows. |
| Status: live breakdown + calibration | Any time; asks nothing. |
| Recheck past rounds with current tester | After changing `backtest.py`. |
| Backtest only (no submit, fast in-process) | Builder trying ideas between rounds. |
| Replay a real round | Re-score the file we actually sent. |

Or in the terminal:

```bash
python round.py
```

## Each round with `round.py`

1. **Points:** for every round marked submitted but not yet scored, it asks for points, then optional platform score, rank and replay score. Enter = not scored yet.
2. **Breakdown:** live table (points, cumulative, vs previous, rank), best/worst round, points by version, backtest calibration and alerts.
3. **Backtest:** if `main.py` or `backtest.py` changed, it asks for a label ("what changed?") and runs. If nothing changed, it reuses the last run.
4. **Submit:** "Submit 'label' as round N?" → `y`. Then submit on the platform.

## Upgrading main.py and the tester off each other

- Every log row carries `main_hash` and `backtest_version` = `0.1-<hash of backtest.py>`, so every edit of either file is tracked automatically.
- **Changed `main.py`?** `round.py` notices and backtests it. Keep it if the 95% CI of the diff vs momentum clears 0 and the leak test passes.
- **Changed `backtest.py`?** Run **Recheck**: it re-runs every *submitted* version (from `runs/`) with the current tester and reports whether it fits our real points better or worse than the tester we had at the time. Keep a tester change only if it explains past rounds better (needs ≥3 different submitted versions to say anything).
- **Need an old version back?** Copy `runs/<run>/main.py` over `main.py`.

## Two PCs: write on one, test on the other

Recommended: **VS Code Live Share** (extension `ms-vsliveshare.vsliveshare`).

- The **tester PC** opens this folder and starts a Live Share session. The **coding PC** joins via the link.
- Both edit the same files live. Everything *runs* on the tester PC: the tester PC presses F5, or the coding PC uses a shared terminal.
- There's only one copy of the files, so nothing goes out of sync.

Alternatives if Live Share doesn't work on the venue Wi-Fi:

- **Shared folder** (OneDrive / Google Drive / a network share): code on PC A, run on PC B. Wait for the sync before running.
- **Git** (private GitHub repo): commit + push on PC A, pull on PC B. Slower per change, but has full history.

Whichever you use: **only the tester PC runs `round.py` / `backtest.py`**. That PC owns `scorekeeper_log.csv` and `runs/`, so two PCs never write the log at the same time.

## 18:00–18:20: fill in the blanks

Search for `FILL` in `backtest.py`, `main.py`, `round.py`, `calibrate.py`. In order of importance:

1. **Metric** → `CONFIG["metric"]`, `higher_is_better`. If they give us a scoring function, paste it into `m_custom` and use `"custom"`.
2. **Target and horizon** → `target`, `horizon`, `execution_delay`, `round_step`; check `realised_outcome()` matches their definition exactly.
3. **Data** → `data_path`, `date_col`, `data_lag`; adapt `load_data()` (pivot if the data is long format). If the bar at t is *not* known when we submit, set `data_lag=1`.
4. **Submission template** → `id_col`, `pred_col`, `pred_range`, `universe_rule` / `template_path`, `to_submission()`. Also fix the asset list in `main.forecast()`.
5. **Missed rounds** → `missed_round_score`, `allow_last_good_fallback`.
6. **Points vs score** → `round.CALIBRATE_ON`: `"points"` (default, ratio calibration) or `"score"` if the platform shows the raw metric.
7. **P&L only** → `cost_bps`, `max_weight`, `allow_short`.

## Reading the output

- **Leak test FAIL** → don't trust the scores; a feature uses future data (usually a missing `.shift(1)` or a `shift(-n)`).
- **Validation FAIL** → the real platform would have rejected that round. Fix the format, never paper over it in the harness.
- **Beats momentum %** and the **95% CI of the paired diff**: a change is only real if the CI is clear of 0. One round is mostly luck.
- **Holdout warning** → probably overfit to the tuning period.
- **ALERT replay ≠ platform** → bug in our scoring or alignment; fix first.
- **k** (in points mode) = points per unit of backtest metric. Stable k = the backtest is a good guide.
