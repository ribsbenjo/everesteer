# Look-ahead bias and leakage

Sources: Fonseca 2026 (arXiv:2607.04958) for the formal theory; the survey (arXiv:2605.19337) §3, §4,
§7, §8 and §13 for applied channels; Na et al. 2026 for agent-input cutoffs.

Contents: 1. The one-sentence rule · 2. Two clocks per datum · 3. Leak catalogue · 4. Formal model
and what it implies · 5. Construct-by-construct rules · 6. Testing: static, dynamic and their limits ·
7. LLM-specific leakage · 8. pandas recipes · 9. Checklist

---

## 1. The one-sentence rule

A pipeline is look-ahead-free if, for every decision epoch t, **changing any datum that becomes
available after t leaves the decision at t unchanged**. This is non-interference: future data is
the "high" input, the decision at t is the "low" output, and the low output must depend on the
low inputs alone.

Why it matters: a backtest reports decisions that were never made. If the decision at t uses
anything knowable only after t, the backtest measures foresight the live system will never have.
Look-ahead is the main way an evaluation flatters a system that later disappoints in deployment.

## 2. Two clocks per datum

Every datum has:
- **reference time**, the instant or period it describes, and
- **availability**, the moment it became knowable to *your* pipeline.

They differ by the ingest lag. Admissibility depends **only on availability**: a value describing
period t−1 that becomes known at t+5 has availability t+5, and using it at t is a leak even though
it is "about the past". The formal discipline is meant to make this confusion impossible to express
by accident.

Availability should be **exogenous and value-independent**: fixed by the source (a publication
date, filing date or bar close) and never computed from data values. Allowing availability to
depend on values ("if price > c, treat this datum as available two bars earlier") is the single
feature that makes look-ahead-freedom undecidable.

## 3. Leak catalogue (from all five papers)

| Leak | Mechanism | Fix |
|---|---|---|
| Full-sample normalization or scaling | mean and std, or a fitted scaler, include the future | fit on past data only (expanding or rolling, or per training fold) |
| Wrong timestamp alignment | feature aligned to reference time, not availability | stamp with availability; add the ingest lag |
| Restatements and vintages | final restated fundamentals instead of as-first-reported | point-in-time / vintage store, `asof(t)` reads |
| Revised macro data | today's revised print used for history | the unrevised value at the release timestamp |
| Survivorship universe | universe built from firms that survived, or from future index constituents | constituents as known at t, including later delistings |
| Execution at the signal close | trading at the same close that generated the signal | next tradable price (next open or VWAP) plus a latency model |
| Post-close aggregation | intraday bars aggregated to a session bar that closes after the decision | causal resample (output bar uses only source bars ≤ its own close) |
| Non-causal join or resample | join across a look-ahead offset; `label`/`closed` wrong; backfill | align sources at a common position; joined row available at the max of its parts |
| Centered moving averages and two-sided filters | the window extends past t | trailing windows only |
| Shuffled CV on time series | random K-fold mixes future into train | chronological splits, walk-forward, embargo around the test window |
| Full-sample regime clustering | the HMM or k-means sees future volatility states | fit regimes online on a t−k lookback |
| Text publication lag | the news timestamp is publication time, not arrival in the pipeline | model ingest lag; embargo |
| Cross-modal asynchrony | stale news paired with the current price | align on availability; report latency |
| LOB feature latency | imbalance at t not computable until t + δ | condition on O_{t−δ} |
| Memory hindsight ("Oracle Fallacy") | a retrieved episode includes its later outcome or narrative | outcome embargo: hide the outcome until t_now ≥ τ |
| Factor library survivorship | the library holds only factors known to have worked | point-in-time library and metadata |
| Embedding leakage | embeddings trained on text with future knowledge | point-in-time corpora |
| Future-annotated event labels | "shock event" labels assigned in hindsight | real-time proxies observable at t (VIX spike, spread widening) |
| Pretraining leakage | the LLM memorized the backtest period's outcomes | post-cutoff evaluation; anonymize tickers and dates; as-of retrieval |
| Retrieval corpus of unbounded vintage | the RAG index includes documents after t | the retrieval call carries `as_of ≤ t` |
| Re-stamping (the insidious one) | a future-derived value is written into an earlier row or given an earlier timestamp | track the dependency, not the label; see §4 |
| Meta-learning task leakage | tasks sampled at random across time | split tasks by time or by disjoint asset classes |
| Search and selection leakage | validation results feed back into factor or strategy search | sealed test set, search budget, deflated statistics |

## 4. Formal model and what it implies (Fonseca)

**Calculus.** A time-indexed value is ⟨v, α⟩, a value plus its availability stamp. The constructs
are: base datum, value projection, pure value operation f, `stamp(e, A)` (re-stamp), `window_k`
(the last k values), `scan` (causal fold), `join` (align sources), `resample_ρ`, `asof(store, A)`
(point-in-time read), `retrieve(ctx, A)` (agentic context fetch as of A) and `decide_t`.

The **availability sublanguage** contains t, constants, av(x), A + δ (a constant lag), and
fixed-arity max and min. It **never reads data values**. A pipeline whose availability terms all
lie in this sublanguage belongs to the decidable fragment, and membership can be checked in linear
time.

**Effects.** A term's effect is a static upper bound on the availability of every base datum whose
value can reach the term:
- pure operation: max of the operand effects;
- join: max (a joined row is available when its *last* component is);
- window and scan: the effect of the newest element (scan: max(seed, current));
- `asof(store, A)` and `retrieve(ctx, A)`: effect A, because only entries with availability ≤ A
  are visible. This is why vintages and restatements remain checkable;
- **`stamp(e, A)`: the effect is still e's effect.** Re-stamping changes the advertised timestamp
  but not what the value depends on. A check that reads only advertised timestamps will accept a
  future value relabelled as past. An effect discipline rejects it. Output-comparing detectors miss
  it as well, because the relabelled value looks stable;
- `decide_t(e)` is accepted only if effect(e) ⊑ t.

**Acceptance also requires:**
- (a) every base series is *availability-monotone*: later positions are never available earlier.
  Assert this at ingestion;
- (b) every resample is *causal*: max ρ←(p) ⪯ ι(p), so an output bar aggregates only source
  positions at or before its own close. For upsampling, each output maps to a unique source
  position ⪯ it;
- (c) every join aligns its sources at a common position, never across a look-ahead offset.

A non-causal resample is "the classic resample leak".

**Boundaries.**
- *Sound:* every accepted pipeline is look-ahead-free.
- *Conservative:* it rejects pipelines whose future dependence cancels (x − x) or is ignored inside
  an opaque operation. These false positives are characterized exactly and can be cleared with a
  local annotation.
- *Undecidable in general:* with value-dependent availability, freedom is Π⁰₁-complete. A leak has
  a finite witness, so detection is semi-decidable, but certifying freedom is not possible.
- The guarantee holds for pipelines expressed in the calculus or faithfully modelled by it, not for
  arbitrary Python.

**Practical design consequence.** Write pipelines so that each input's availability is a
declared function of the schedule (bar close, release calendar, filing date + lag). Express the
pipeline with windows, scans, joins, causal resamples and as-of reads, and never derive timing from
values. Then even a simple structural review gives strong guarantees.

## 5. Construct-by-construct rules

- **Rolling window** of length k at position p covers [p−k+1, p] and never goes past p. Near the
  start of the series it is a truncated prefix (warm-up). Treat warm-up rows as NaN, not as filled
  values.
- **Expanding statistics** are causal by construction. Full-sample `.mean()` and `.std()` are not.
- **Cumulative folds** (`cumsum`, EWMs computed forward) are causal. Anything computed backward
  from the end (`bfill`, reversed EWM, `shift(-k)`) is not.
- **Joins** of sources with different update cadences take the latest availability among the
  inputs. Using `merge_asof` with `direction="backward"` on availability timestamps is causal;
  `"forward"` and `"nearest"` are not.
- **Resampling:** make sure the output label sits at the *end* of the interval and that the
  interval contains only source rows ≤ that label. Labelling a daily bar with its start time while
  it contains the whole day's data leaks it into the start-of-day decision.
- **Point-in-time stores** keep every version with its availability timestamp and are read with
  `as_of = t`.

## 6. Testing

**Static (preferred where possible).** Declare an availability per input and check the dependency
structure. `scripts/leak_check.py lint` flags code patterns that often break causality: negative
shifts, `center=True`, `bfill`/`backfill`, `interpolate`, full-sample `fit`/`fit_transform`,
`rank`/`qcut` over the time axis, `merge_asof(direction="forward"/"nearest")`, and `resample`
calls whose `label`/`closed` need checking. Its hits are prompts for review, not verdicts.

**Dynamic two-run oracle.** At a sampled epoch t, compute the decision on the stored data, then
perturb *every* datum with availability > t and recompute. If the decision changes, you have a
witnessed leak. `scripts/leak_check.py oracle` runs this over many epochs with several kinds of
perturbation, and also runs a truncation test (`build(df[:t]).loc[t] == build(df).loc[t]`).

**Where dynamic tests fail.** On Fonseca's corpus, the two-run detector missed 54.5% of leaks and
the tiling detector 100%:
- *coverage miss*: the leak sits at an epoch the tester didn't sample, or (as in the weekly-release
  case) the lag is short enough that sampled epochs never flip the selected row. Use many epochs,
  including every epoch near release dates;
- *perturbation-insensitivity miss*: the future value passes through a threshold that the default
  perturbation doesn't cross. Use several perturbation types (sign flip, large shocks, shuffle,
  NaN);
- *re-stamp miss*: a future-derived value carries an earlier, admissible-looking timestamp. An
  oracle decides what counts as "future" from those timestamps, so it never perturbs the value
  and both runs agree. In pandas this most often means the *input* is stamped by reference time
  rather than availability (fundamentals at period end, restated values, revised macro, a survivor
  universe), or a future-derived column was precomputed and saved into the dataset. Only
  dependency or availability tracking catches this: declare and verify each input's availability.

A passing dynamic test is necessary, not sufficient. Silence proves nothing.

**Worked demonstration with the bundled script** (eight synthetic price series, 750 days, 12 epochs):

| Pipeline | 6 perturbation kinds | same, data with 5% gaps | noise only | truncation | lint |
|---|---|---|---|---|---|
| trailing momentum, cross-sectional rank | pass | pass | pass | pass | clean |
| expanding z-score | pass | pass | pass | pass | clean |
| `shift(-1)` future return | **leak** | **leak** | **leak** | **leak** | high |
| centered rolling mean | **leak** | **leak** | **leak** | **leak** | high |
| full-sample z-score | **leak** | **leak** | **leak** | **leak** | review |
| `bfill()` | pass | **leak** | pass | pass | high |
| future max passed through a threshold | **leak** (extreme only) | **leak** | pass | pass | none |

The backfill leak shows up only when the data has gaps, which is a data-dependent coverage miss.
The threshold leak is caught only by the ±1e9 perturbation, which is a perturbation-insensitivity
miss. A single noise perturbation over a handful of epochs, as many backtest harnesses use, would
pass both. Combine lint, several perturbation kinds, gappy test data and many epochs.

**Contamination check (survey).** If performance drops sharply when you move from a random split to
a strict time-consistent split, the model was contaminated.

## 7. LLM-specific leakage

- **Pretraining memory.** The model may "know" what happened after t for any historical t inside its
  training window, so clean code doesn't guarantee a clean experiment. Use one or more of:
  evaluation strictly after the model's knowledge cutoff (Na et al. tested only on post-cutoff 2026
  Q1 data); anonymization of tickers, company names and dates (Jeon & Lee 2026, "blindfolded"
  LLMs); and claim-level contamination attribution (Shapley-based, Zhang, Chen & Stadie 2026).
- **Tools and retrieval** must carry an explicit `current_date`/`as_of` cutoff (Na et al.) and
  return only items with availability ≤ t. A retrieval step over a corpus of unbounded vintage
  should be rejected.
- **Agent memory** needs an outcome embargo. Post-trade lessons stored in memory must not reach
  decisions dated before the lesson existed.
- **Summaries** can hallucinate or import future facts. Keep hash pointers to raw logs and audit the
  summaries.

## 8. pandas recipes

```python
# Shift features so row t uses data knowable at t (when bar t itself isn't known at submission time)
feat = raw_feature.shift(1)

# Causal normalization: expanding or rolling, never the full sample
z = (x - x.expanding(min_periods=60).mean()) / x.expanding(min_periods=60).std()
z_roll = (x - x.rolling(60).mean()) / x.rolling(60).std()

# Cross-sectional ranks are fine (axis=1 = across assets on the same row); ranks over time are not
cs_rank = panel.rank(axis=1, pct=True) - 0.5

# Join slow data (e.g. fundamentals) on availability, never on period end
fund = fund.assign(available_at=fund["filing_date"] + pd.Timedelta(days=1))
merged = pd.merge_asof(prices.sort_values("date"), fund.sort_values("available_at"),
                       left_on="date", right_on="available_at", by="ticker",
                       direction="backward")

# Causal daily resample from intraday: label and close on the right edge
daily = intraday.resample("1D", label="right", closed="right").last()

# Regime label fitted online, not on the full sample
vol = ret.rolling(20).std()
regime = (vol > vol.expanding(min_periods=250).quantile(0.8)).astype(int)
```

## 9. Checklist

1. Every input has a written availability rule, and nothing's availability depends on data values.
2. Every transform is trailing, expanding or fitted per fold. Nothing uses full-sample statistics,
   centered windows, backfill or forward-looking joins.
3. Joins take the max availability, and resamples are causal.
4. Point-in-time data is used for fundamentals, macro, the universe and index membership.
5. Execution happens at the next tradable price, not the signal close.
6. Splits are chronological, with an embargo around the test window and a sealed holdout for any
   search.
7. LLM components are evaluated after their cutoff or anonymized. Retrieval and tools are as-of t.
   Memory has an outcome embargo.
8. `leak_check.py lint` has been read through, and `leak_check.py oracle` passes with many epochs
   and several perturbation types.
9. You have noted what the dynamic test can't see (re-stamping, unsampled epochs, weights).
