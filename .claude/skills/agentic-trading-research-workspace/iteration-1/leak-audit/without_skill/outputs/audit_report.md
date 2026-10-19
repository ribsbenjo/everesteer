# Look-ahead audit of `features.py`

**Setup.** We submit after the close of day t and are scored on the return from close t to close t+1. So a feature row dated t may use anything known at the close of t, and nothing after it.

**Result.** I found 8 problems. Every feature column except `cs_move` was affected, and the ticker universe was too. `features_fixed.py` has the same `build_features(prices, fundamentals)` signature and the same output columns, and it passes the truncation test on all 599 cutoff dates with zero differences. The original `features.py` is unchanged.

## Problems found

| # | Line in `features.py` | What leaks | Fix in `features_fixed.py` |
|---|---|---|---|
| 1 | `prices.dropna(axis=1)` | Survivorship. A ticker is kept only if it has no gaps over the **whole** sample, and you can't know that until the sample ends. It dropped 6 of 30 tickers: T07 (delists 2023-09-22), T03 (lists 2022-08-01) and T11, T15, T22, T27, which each have six one-day gaps. A model trained this way never sees a stock that later dies. | Keep every ticker. A ticker gets a row on date t only if it has a usable close at t. |
| 2 | `prices.bfill()` | A missing close is filled with the **next** day's close. All 24 internal gaps in the sample got tomorrow's price. Before IPO, the price was also back-filled to the first print. Problem 1 currently hides this, but it goes live as soon as anyone removes the `dropna`. | `ffill(limit=5)`: carry the last known close forward for at most 5 days. |
| 3 | `(mom - mom.mean()) / mom.std()` | The z-score uses each ticker's full-sample mean and standard deviation, which include future data. | Expanding mean and std up to and including t (60 observations minimum). |
| 4 | `rev.rank(pct=True)` | `DataFrame.rank` defaults to `axis=0`, so this ranks each ticker's value against its **entire history, future included**. | `rank(axis=1, pct=True)`: a percentile across tickers on the same day. |
| 5 | `rolling(21, center=True).std()` | A centred window uses the next 10 days of returns. | Trailing `rolling(21).std()`. |
| 6 | `mkt_vol.quantile(0.8)` | The regime threshold is the 80th percentile of the full sample. | Expanding 80th percentile up to t (120 observations minimum). NaN during warm-up instead of a silent 0. |
| 7 | `weekly.reindex(..., method="bfill")` | Weekly bars are labelled with the Friday that ends the week. Back-filling gives Monday through Thursday the coming Friday's 4-week return, so up to 4 days of future prices. | `method="ffill"`: each day gets the last completed week. A Friday row uses that Friday's close, which is known at the close. |
| 8 | `merge_asof(..., right_on="period_end")` | EPS was matched on the quarter end, but it only becomes public on the filing date, 25 to 49 days later (37 days median). For example, T00 on 2022-01-03 already "knew" Q4-2021 EPS, which wasn't filed until 2022-01-27. | Match on `filing_date` and use it from the next trading day (`allow_exact_matches=False`). A late filing for an older quarter never overwrites a newer one. |

`cs_move` is fine: a cross-sectional rank of today's return is known at today's close. In the original file its values still came out wrong, because the cross-section was the survivorship-filtered universe from problem 1. Test 1b below shows its own formula is clean.

## How I checked the fix (`leak_check.py`, reproducible)

1. **Truncation test, every date.** For each cutoff T (599 dates), rebuild the features from only what is known at the close of T: prices up to T, and filings dated strictly before T. Then compare row T with the full-sample run. Any difference means row T used future data.
2. **Perturbation test.** Scramble everything after T (random-walk the prices, delete 5% of future prints, randomise EPS filed on or after T). Then check that every row dated T or earlier is unchanged, including which rows exist. I used 5 cutoffs.
3. **One fix at a time.** I applied the fixes cumulatively and reran test 1 after each one. From step 3 on, each fix drove its own column to zero and left the others unchanged. Steps 1 and 2 change counts elsewhere because more tickers are being compared (table below).
4. **Smell test.** I computed the mean daily Spearman IC against the actual target. Leaked features look too good to be true.

Mismatched cells in the truncation test, all 30 tickers, 599 cutoffs (`only_a`/`only_b` = rows present in only the full-sample or only the live run):

| applied so far | only_a | only_b | mom_z | rev_pct | vol | weekly_mom | cs_move | high_vol | eps / ey |
|---|---|---|---|---|---|---|---|---|---|
| original | 0 | 876 | 13 896 | 14 225 | 13 896 | 11 136 | 10 174 | 1 008 | 6 040 |
| + keep all tickers (1) | 173 | 0 | 17 067 | 17 465 | 17 060 | 13 680 | 4 630 | 1 596 | 7 367 |
| + ffill instead of bfill (2) | 0 | 0 | 17 075 | 17 487 | 17 075 | 13 673 | **0** | 1 388 | 7 378 |
| + expanding z-score (3) | 0 | 0 | **0** | 17 487 | 17 075 | 13 673 | 0 | 1 388 | 7 378 |
| + cross-sectional rank (4) | 0 | 0 | 0 | **0** | 17 075 | 13 673 | 0 | 1 388 | 7 378 |
| + trailing vol (5) | 0 | 0 | 0 | 0 | **0** | 13 673 | 0 | 1 388 | 7 378 |
| + expanding quantile (6) | 0 | 0 | 0 | 0 | 0 | 13 673 | 0 | **0** | 7 378 |
| + weekly ffill (7) | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 7 378 |
| + EPS on filing date (8) = `features_fixed.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

The "+ keep all tickers" row shows the hidden `bfill` leak becoming active: 173 rows, mostly T03 before its IPO, and 4 630 `cs_move` cells. The final configuration gives exactly the same values as `features_fixed.py`.

- **Test 1b.** I reran the truncation test on the 24 complete-history tickers, so the universe can't change. The original shows the same per-feature leaks and **0** for `cs_move`. The fixed file shows 0 everywhere.
- **Perturbation test.** The fixed file shows 0 differences at all 5 cutoffs. In the original, deleting future prints made whole tickers vanish from the past: between 2 424 and 12 024 rows dated on or before T disappeared.
- **IC smell test.** Original `weekly_mom` had IC 0.166 (t = 18.6) and original `vol` had IC 0.153 against |next-day return| (t = 16.9). Fixed: -0.003 (t = -0.4) and -0.006 (t = -0.7). Every fixed feature has |t| < 1.4. The leaks in `mom_z`, `rev_pct` and EPS don't show up in IC on this sample data, which is why the truncation test is the real proof.

## Assumptions and behaviour changes

- **Timing.** CLAUDE.md asks for every feature to be shifted one period. Here that shift is built into the alignment: row t holds data through close t, and the target is the t to t+1 return, which `forward_return(prices)` returns on row t. If your training code stores the target on the date the return is realised (t+1), shift the features by one row per ticker instead. Do one or the other, never both and never neither.
- **Filings.** Filings carry no time of day, so I assume one dated D could come out after the close. Its EPS is used from the next trading day.
- **Interpretation.** `rev_pct` is now a cross-sectional percentile and `mom_z` an expanding per-ticker z-score. These are my reading of the intent; a cross-sectional z-score for `mom_z` would be equally leak-free.
- **More rows.** The output has 17 705 rows instead of 14 400: all 30 tickers while listed, plus 5 stale-price rows for T07 after delisting. Their target is NaN, so they drop out of training. Only submit tickers in the official universe.
- **Warm-up NaNs.** `mom_z` is NaN for the first 79 days and `high_vol` (now float) for the first 139. If the model can't take NaN, drop those rows or fill with a neutral value. Never backfill.
- **Run the checks.** Run `python3 leak_check.py` in this folder (about a minute). It needs `features.py`, `features_fixed.py` and the two CSVs next to it.
