# Everesteer hackathon — target_everest models (models-v1)

Code for the Everesteer futures hackathon (event dataset: 178 binned features, `-1` = missing,
`target_everest` quintile-binned, ids re-keyed every exped). Scoring is the event round score
`S = FIT20 + 3*UNQ + INOV - (benchmark FIT20 + benchmark INOV)`, paid as `b*arctan(3S/b)`, `b = 0.63662`.

## Models

All models work per exped (cross-section): z-score every feature within the exped (missing -> 0),
take a weighted sum, then partially neutralise against a LightGBM mimic of the `v1_sherpa` benchmark
(the real benchmark is withheld for event keys) and rank to [0, 1].

| Model | Slot name (server-assigned) | Method | Builder |
|---|---|---|---|
| M1 | fermat-lazy-margin | all 178 features signed by mean per-exped corr with `target_everest` over the last 100 train expeds; neutralised 0.5 vs sherpa mimic | `lib/build_model1.py 100 0.5 model1` |
| M2 | kolmogorov-caffeinated-drift | as M1, neutralised 1.0 | `lib/build_model1.py 100 1.0 model2` |
| M3 | cover-asymptotic-drift | ridge (lambda 3e5) on per-exped z-scored features, last 50 train expeds; neutralised 0.5 | `lib/build_linear.py model3 0.5 lib/w_model3.npy` |
| M4 | fama-reckless-martingale | M3 + 50-exped signed sum for `target_Tinzar`, equal variance blend; neutralised 0.5 | `lib/build_linear.py model4 0.5 lib/w_model4.npy` |
| M5 | ramanujan-paranoid-margin (2nd run) | M1 also neutralised 0.5 against the guessed core-feature average | `lib/build_linear2.py model5 0.5 lib/w_model1.npy lib/core_features_guess.json 0.5` |
| M6 | ramanujan-paranoid-margin | M1 + 50-exped signed sums for `target_Azgour`, `target_Tinzar`, `target_Moussa`, equal variance blend; neutralised 0.5 | `lib/build_linear.py model6 0.5 lib/w_model6.npy` |

Weight vectors (`lib/w_*.npy`) are produced by the snippets in the session log; `lib/exp1.py`-`exp5.py`
show how each was derived from `lib/ex_corr.npy` / `lib/excorr_targets.pkl` (per-exped feature-target correlations).

### Offline fold scores

Three time-ordered folds: 300-exped test windows ending at 60 %, 80 % and 100 % of train, with a 40-exped gap
after the training data. `S_raw = FIT + 3*UNQ + INOV` (UNQ vs the real `v1_sherpa`, INOV vs the guessed core set).
Full per-fold FIT/UNQ/INOV in `final_fold_table.csv`.

| Model | f60 | f80 | f100 | mean | min |
|---|---|---|---|---|---|
| M1 | -0.0048 | -0.0009 | +0.0119 | +0.0021 | -0.0048 |
| M2 | -0.0040 | -0.0003 | +0.0105 | +0.0021 | -0.0040 |
| M3 | -0.0147 | +0.0032 | +0.0281 | +0.0055 | -0.0147 |
| M4 | -0.0114 | +0.0171 | +0.0369 | +0.0142 | -0.0114 |
| M5 | -0.0021 | -0.0035 | +0.0065 | +0.0003 | -0.0021 |
| M6 | +0.0032 | +0.0202 | +0.0330 | +0.0188 | +0.0032 |

### Practice board (validation split, 175 expeds, 2026-10-19)

| Model | FIT | UNQ | INOV | round score | rank (of 1068) |
|---|---|---|---|---|---|
| M6 | 0.0278 | 0.0254 | 0.0263 | 0.2812 | 1 |
| M1 | 0.0272 | 0.0246 | 0.0246 | 0.2697 | 2 |
| M5 | 0.0271 | 0.0246 | 0.0247 | 0.2696 | (same slot as M6) |
| M2 | 0.0247 | 0.0246 | 0.0224 | 0.2579 | 5 |
| M4 | 0.0250 | 0.0230 | 0.0247 | 0.2522 | 8 |
| M3 | 0.0247 | 0.0229 | 0.0245 | 0.2503 | ~10 |

Findings: ids are re-keyed per exped (no instrument history); interleaved CV leaks through the 20-exped target
overlap; feature-target relationships drift strongly, and every model fit on long history is negative on the last
fold; `v1_sherpa` looks in-sample on train. Short recent windows and auxiliary-target blends were the only robust
settings offline.

## Environment

```bash
pip install "everesteer-api[scoring]" lightgbm cloudpickle pyarrow scipy pandas numpy
export EIQ_API_KEY=...        # your event-scoped key
export EIQ_BASE_URL=...       # API base URL (omit for the default)
```

Never commit keys. The SDK reads both variables automatically.

Download data (SDK, authenticated via the env vars):

```bash
python3 -c 'from everestapi import EverestAPI; api = EverestAPI(); [print(api.download_dataset(universe="futures", split=s, output_path=f"futures_{s}.parquet")) for s in ("train","validation","train_benchmark_models")]'
```

`lib/eiqlib.py` expects the parquet files in `/home/climber/work/` (edit `load()` for another path). Then build the
sherpa mimic and feature correlations with `lib/exp1.py` (writes `lib/Z.npy`, `lib/ex_corr.npy`) and `lib/exp2.py`
(writes `lib/excorr_targets.pkl`).

## Running a round

1. Build the model pickles with the builder commands above (each writes `models/<name>.pkl`).
2. When a round is open (`get_started` shows `open_window` set and `live_data_available: true`):

```bash
python3 lib/round_predict.py round_2 model6 model1
```

   This downloads the open round's live split to `rounds/round_2/live.parquet` and writes
   `rounds/round_2/<model>_preds.parquet` for each model. It does not submit.
   `lib/round_wait.py` does the same but polls until the round opens.
3. Submit one model per call on the event lane:

```bash
python3 lib/round_submit.py round_2 model6 ramanujan-paranoid-margin "round2 M6"
```

4. Draft stakes with `set_stake_allocation` (MCP) while the round is open; they lock at round close.

Practice-board submissions use `lib/submit.py <slot> <preds.parquet> <model.pkl> <label>`.
