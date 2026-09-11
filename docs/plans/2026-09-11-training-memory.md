# Plan — lower the training peak memory, same model (api_v3), 2026-09-11

**Status: APPROVED 2026-09-11 ("den Plan umsetzen"); implemented on branch
`feat/training-memory`.** Later the same day the owner settled the open questions and
widened the scope: `auto` as an explicit value, the head-fit threads shown during a
run and counted in the time estimate, the test container's budget at 6000 MB, and
Phase 4 in scope and measured — see "Decisions" at the end and tasks 1.6–1.8 and
Phase 4. Companion to `docs/plans/2026-09-08-ergaenzungsplan.md` (same conventions).

## Goal

A training run must fit into a container with a fixed memory limit — the concrete case is
the 8 GB `apiv3-test` container and the full WLO export `wlo_max_dataset.csv.gz`
(426,724 records, 148 columns), whose `auto` run was OOM-killed on 2026-09-11 at
6.98 GB of anonymous RSS while still growing — **without changing a single number the
model produces**. Every item below is output-identical by construction: vocabulary, idf
weights, feature matrix, coefficients, thresholds and metrics stay bit-identical to what
today's code writes. Quality is not traded; wall-clock is, and only where a run would
otherwise die.

## Where the memory goes — measured, not assumed

Measured 2026-09-11 on this machine (Windows, Python 3.12, scikit-learn 1.6.1, scipy
1.13.1) with a 5 ms RSS sampler around each step (`scratchpad/measure_vec_memory.py`,
to become `scripts/benchmark_training_memory.py` in task 0.3). "Peak" is the transient
maximum above the RSS before the step, "retained" what is still held after the step's
objects were deleted and `gc.collect()` ran.

**Provisional.** Two measurement runs overlapped on this machine while these tables
were taken, so every wall-clock figure below is indicative only (the RSS figures are
per process and unaffected). And all steps shared one process, so a later step could
reuse memory an earlier one left behind — the sequential rows were measured right after
the concurrent ones and may read low. Task 0.3 re-measures every step in a fresh
process; its tables are the baseline the gates compare against.

### data_30k_ai.csv — 26,450 rows × 48 labels, ø 616 chars, 16 MB of text (UTF-8)

| Step (today's code path) | Peak | Retained | Result it produces |
|---|---:|---:|---|
| `load_dataset` (read + combine + clean) | +121 MB | +52 MB | 22 MB of `str` objects |
| word TF-IDF fit alone (cap 80k) | +245 MB | +129 MB | 12.9 MB matrix; vocabulary 843,605 → 80,000 terms |
| char TF-IDF fit alone (cap 120k) | +179 MB | +75 MB | 43.1 MB matrix; vocabulary 358,974 → 120,000 terms |
| `TfidfBackend.fit_transform` — word ∥ char in 2 threads (production) | **+330 MB** (16.7 s) | +209 MB | 56 MB matrix |
| same, word *then* char (sequential) | +208 MB (17.2 s) | +101 MB | identical matrix |
| one head fit, C=8, 6 threads, newton-cg | **+645 MB** | +21 MB | 37 MB of float32 coefficients |

### data_300k.csv, first 100,000 usable rows — 59 labels, ø 728 chars, 70 MB of text

| Step | Peak | Retained | Result |
|---|---:|---:|---|
| `load_dataset` (reads the whole 340,630-record file, 558 MB) | **+1,481 MB** | +275 MB | 112 MB of `str` objects |
| word fit alone | +798 MB | +357 MB | 64 MB matrix; vocabulary 2,654,836 → 80,000 |
| char fit alone | +715 MB | +267 MB | 197 MB matrix; vocabulary 1,106,905 → 120,000 |
| `TfidfBackend.fit_transform` (2 threads, production) | **+1,243 MB** (81.8 s) | +643 MB | 260 MB matrix (340 non-zeros/row) |
| same, sequential | +869 MB (100.6 s) | +350 MB | identical matrix |
| one head fit, C=8, 6 threads | **+2,871 MB** (= 11 × the matrix) | −128 MB | 45 MB of coefficients |

Head-fit peak as a function of the thread count (30k, the same 56 MB matrix, C=8):

| threads | peak | per thread |
|---:|---:|---:|
| 1 | +142 MB | 2.5 × matrix |
| 2 | +279 MB | 2.5 × matrix |
| 4 | +459 MB | 2.1 × matrix |
| 6 | +645 / +679 / +680 MB (three runs) | 1.9–2.0 × matrix |

Linear in the thread count; 1.8–2.5 × the matrix per concurrent fit. Retained after the
fit: 20–50 MB — the head fit leaves almost no residue, unlike the vectorizer.

### Baseline (task 0.3) — every step in a fresh process

`scripts/benchmark_training_memory.py`, 2026-09-11, before any change to the pipeline
(`vectorize` and `vectorize-before` were still the same concurrent code). This is what
the gates compare against.

| Step | data_30k_ai.csv (26,450 rows) | data_300k.csv (100k of 156,174 usable rows) |
|---|---:|---:|
| load (whole file) | +116 MB peak, +47 MB kept, 1.6 s | +1,476 MB peak, +422 MB kept, 21.4 s |
| vectorize (word ∥ char) | +354 MB, 14.8 s (56 MB matrix) | +1,281 MB, 74.8 s (260 MB matrix) |
| head fit C=8, 6 threads | +627 MB (11.2 × matrix), 19.2 s | +3,103 MB (11.9 × matrix), 120.5 s |
| head fit, ThreadBudget of RSS + 3 × matrix | 1 thread, +143 MB, 60.4 s — within budget | 1 thread, +572 MB, 387 s — within budget |

In fresh processes the vectorizer leaves little behind (+26 / +52 MB retained): the
"residue" of the first tables was largely memory one step freed and the next reused.

### The five causes, with the code that causes them

1. **The head fit copies the matrix once per concurrent thread.** scikit-learn's
   newton-cg path materialises `hX = diag(h) @ X` for every label and every Newton
   iteration (`sklearn/linear_model/_linear_loss.py:728-731`), and the previous
   iteration's copy is still referenced while the next one is built
   (`sklearn/utils/optimize.py:276`). Under the `threading` backend `n_jobs` threads
   therefore hold `n_jobs × (1.8–2.5) × matrix` in flight. Measured: +645 MB for a
   56 MB matrix at 6 threads, +2,871 MB for a 260 MB matrix at 6 threads, and a peak that
   grows linearly with the thread count (table above). The README's "all cores share one
   matrix, ~1× RAM" describes the
   *input* matrix only; the working memory of the fits scales with the thread count.
   This is the largest single term at production size and the likeliest killer of the
   wlo_max run (see the extrapolation below).
2. **The vectorizer fit builds everything before it prunes.** `CountVectorizer._count_vocab`
   (`sklearn/feature_extraction/text.py:1246-1309`) keeps a dict of *every* n-gram
   (843k / 2.65M word terms for 80k kept) plus per-document index lists and a count
   matrix over the whole vocabulary; `_sort_features` then builds a sorted list of
   (term, index) tuples and `_limit_features` column-slices a second matrix. Peak 4–19×
   the final matrix. `stop_words_` is *not* involved — sklearn 1.6 no longer stores it
   (checked; the skops dump of a fitted vectorizer is 0.8 MB).
3. **Word and char vectorizers run concurrently**, so their peaks add
   (`app/vectorizers.py:93-96`). Measured gain of the concurrency: none at 30k
   (16.7 s vs 17.2 s), 19 % at 100k (81.8 s vs 100.6 s) — the analyzer loop holds the
   GIL. Cost: +59 % (30k) / +43 % (100k) peak.
4. **Loading holds three full-length copies of the text.** `data.load_dataset`
   (`app/data.py:232-234`) reads the whole frame, `combine_text_columns` builds a second
   full-length Series and `_clean_in_chunks` a third; only the cleaned list survives.
   Peak ≈ 7.6× the UTF-8 text it keeps on the 30k file (+121 MB for 16 MB). The 100k
   row is not a ratio: its load read the whole 558 MB file (340,630 records) while the
   retained figure covers only the 100k subset kept afterwards.
5. **Residue.** After each step 100–650 MB stay resident (pymalloc arenas and heap
   fragmentation: the "retained" column). It stacks under the next step's peak, and
   across runs it is the back-to-back thrash already documented in July.

Smaller, fixed terms: out-of-fold buffers `rows × labels × 4 B` per C candidate
(3 × 100 MB at 426k rows × 59 labels); the int8 label matrix (25 MB).

### Extrapolation to the killed run (estimate — the per-row constants above, not a measurement)

wlo_max: 426,724 records; the usable share after the discipline filter is unknown, so
take 300k usable rows as the working assumption. Constants from the 100k run: 2.6 KB
of matrix per row, 12.4 KB of vectorizer peak per row (concurrent), 2.75 KB retained per
loaded row, load peak ≈ 2.7× the plain CSV size.

| Phase of an `auto` run (3-fold CV) | Estimate |
|---|---:|
| Load: transient peak (1.4 GB plain CSV) → retained texts + labels | ~3.7 GB → ~0.8 GB |
| Fold 1 vectorization (200k train rows), concurrent | +2.5 GB transient, ~1.2 GB residue |
| Fold matrix (train 200k + test 100k rows) | ~0.8 GB |
| First head fit, C=2, **9 threads** (`effective_n_jobs` at 60 % of 16 cores) | 9 × ~2 × 0.52 GB ≈ **+9 GB** |

Base (~2–2.5 GB) plus the first head fit (~9 GB) is where 7.4 GB of VM memory runs out —
consistent with the observed death, and with the phase not being visible in the log
(the head fits emit no log line of their own). Cutting the vectorizer and loader peaks
alone would **not** have saved this run; bounding the head-fit threads would.

## Approaches considered

| | Approach | Verdict |
|---|---|---|
| A | Tune it away by config: lower `APIV3_N_JOBS`, lower the feature caps, `fast` profile | Caps and profile change the model; `n_jobs` is a manual guess per dataset and machine, and nothing stops the next run from dying. Rejected as the *fix*, fine as a stop-gap (see "Until then"). |
| **B** | **Output-identical engineering** (chosen): bound the head-fit threads by a memory budget; fit the two vectorizers one after the other; build the vocabulary in two passes so nothing beyond the kept terms is ever materialised; stream the CSV; release phase objects explicitly; make the peak visible. | Every step verified bit-identical; costs wall-clock only where the alternative is an OOM. |
| C | `HashingVectorizer` (no vocabulary, constant memory) | Hash collisions change the features; no `vocabulary.json`, no explain, bundle format 3. Rejected. |
| D | Another solver: `lbfgs` needs no `hX` but upcasts X to float64 per fit (same order of memory) and converges to a different optimum; `saga` is slow on high-dimensional TF-IDF and also changes the optimum. | Rejected — not "the same model". |
| E | Run the fitting in a child process | Does not lower the peak of one run; it removes the residue between runs, returns memory to the OS, and turns an OOM kill of the run into a job error instead of a dead API. Phase 4 (in scope since the owner's decision). |

## Global constraints (from CLAUDE.md, unchanged)

13 runtime dependencies stay 13 (the RSS probe is `ctypes`/`/proc`, not `psutil`);
files ≤ ~300 lines, split by responsibility; test-first; bundle format stays 2; the
single-worker design stays; English comments explaining *why*.

## Design

### New module `app/memory.py` (stdlib only, ~150 lines)

```python
def rss_bytes() -> int
    # Linux: /proc/self/statm; Windows: psapi.GetProcessMemoryInfo via ctypes
    # (GetCurrentProcess.restype = c_void_p — the pseudo-handle is truncated otherwise).
    # Never raises: 0 when unavailable (macOS: resource.getrusage only knows the
    # lifetime PEAK, which is not a current reading).
def memory_limit_bytes(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int | None
    # cgroup v2 memory.max ("max" = None), else v1 memory/memory.limit_in_bytes
    # (>= 2**60 = None). Mirrors settings._cgroup_cpu_quota.
def matrix_bytes(matrix) -> int   # CSR/CSC: data + indices + indptr; dense: nbytes
class PeakSampler:                # context manager; a daemon thread samples rss_bytes()
    peak_bytes: int               # every 0.5 s — the head fit's copies live inside ONE
                                  # call, where no progress callback ever looks
class ThreadBudget:               # one per training run
    requested: int                # settings.effective_n_jobs()
    budget_bytes: int | None      # None = uncapped (today's behaviour)
    chosen: list[int]             # what each fit got — reported in the bundle
    def for_matrix(self, matrix) -> int:
        # min(requested, max(1, (budget - rss_bytes()) // (2.5 * matrix_bytes(matrix))))
        # 2.5 = FIT_COPIES_PER_THREAD: the measured 1.8-2.5x, at the conservative end
```

`Settings` gains `train_memory_mb: int | None = None` (`APIV3_TRAIN_MEMORY_MB`):
`None` → the cgroup limit minus 15 % headroom when a limit exists, otherwise uncapped;
`0` → uncapped explicitly. `effective_train_memory_bytes()` resolves it; `GET /config`
reports it next to `effective_n_jobs`. The per-thread factor is a constant with a
comment pointing at the measurement, not a setting: nobody should have to know it.

### Wiring the budget (Phase 1)

`deploy.fit_evaluate_deploy` builds one `ThreadBudget` and hands it to `select_c`,
`cross_val_evaluate` and the deploy fit as `thread_budget: ThreadBudget | None`
(keyword-only; `None` keeps the plain `n_jobs` so tests and the benchmark scripts are
untouched). Each fit calls `make_head(c, n_jobs=budget.for_matrix(x_tr), ...)` — the
matrix it is about to fit on is exactly what the copies scale with. The chosen thread
count travels into the progress detail (`"C=2 (1/3) — 4 threads"`) so a slowed run
explains itself.

### Vectorizer (Phase 1 → Phase 2)

Phase 1: `TfidfBackend.fit_transform`/`fit` run word then char (drop the
`ThreadPoolExecutor`). Phase 2: new module `app/vocabulary.py` (~150 lines) replaces
`TfidfVectorizer.fit_transform` for the *fitting* corpus:

```python
def count_terms(analyze, texts) -> tuple[dict[str, int], np.ndarray, np.ndarray]
    # term -> first-seen id, plus df and tf per id (int64). No per-document lists.
def select_terms(index, dfs, tfs, n_docs, *, min_df, max_df, max_features)
    # alphabetical order, then sklearn's exact mask + (-tfs[mask]).argsort()[:limit]
    # (tfs as float32, like the count matrix it sums) -> kept terms in column order
    # and each one's first-seen id
def fit_transform_exact(vec: TfidfVectorizer, texts, chunk_rows=20_000) -> csr_matrix
    # pass 1 + select; pass 2 counts the kept terms chunk by chunk (a CountVectorizer
    # with a fixed vocabulary), TfidfTransformer.fit/transform on that count matrix,
    # vec.vocabulary_ + vec.idf_ (public setter) -> the same fitted state and matrix
def transform_chunked(vec, texts, chunk_rows=20_000)  # vstack of per-chunk transforms
```

"The same matrix" is meant array for array, not only value for value. The reference
builds its count matrix with columns in *first-seen* order, sorts each row by that
order, and only then renumbers the columns alphabetically — so every row of the training
matrix stores its entries in first-seen order. A sparse `X @ w` sums a row in stored
order, so a matrix with the same values in sorted order would give the solver
last-bit-different sums and, over Newton iterations, not quite the same coefficients.
Pass 2 therefore counts with the kept terms numbered by first-seen rank and renumbers
afterwards, exactly like `_sort_features` does; the tests compare `data`, `indices`
and `indptr` with `array_equal`, not only `(ref != x).nnz`.

Two guards keep "exact" honest. The reference ranks terms by float32 column sums,
which are exact integers only below 2**24 (16.7M) occurrences of one term; beyond that
the fit falls back to the reference `fit_transform` (the benchmark reports the largest
count, so how far a dataset is from the limit is a number, not a guess). Configurations the codebase never builds
(`binary`, `use_idf=False`, a preset `vocabulary`) fall back the same way.

Prototype `scratchpad/proto_two_pass.py`, 4,000 real texts, word (1,2) and char_wb (5,5):
vocabulary, `idf_` (values *and* float32 dtype), the training matrix and the transform of
500 unseen texts are **identical** to `TfidfVectorizer.fit_transform` — `(ref != x).nnz
== 0` and `array_equal` on the dense forms. Time 1.25× (word) / 1.4× (char) of the
reference on that sample. What the two passes never build: the per-document index
lists and count matrix over 843k–2.65M terms and their sorted tuple list; the dict of
term counts remains (the one cost that is inherent to exact pruning). The persisted
bundle does not change: `vocabulary.json` + `idf_` inside `vectorizer.skops`, exactly as
`model_io._split_vocabularies` writes today, and `explain.py` never sees a difference.
`transform` of validation/test/fold rows goes through `transform_chunked` too, so the
per-document lists of `_count_vocab` are bounded by the chunk instead of the split.

### Loader (Phase 3)

`data.py` is at 335 lines; the loader moves first (behaviour-preserving) into
`app/dataset_load.py` (`LoadedData`, `combine_text_columns`, `_clean_in_chunks`,
`load_dataset`; `data.py` keeps clean/labels/split). Then `load_dataset` reads with
`pd.read_csv(..., chunksize=50_000)` and does combine → clean → split labels → filter
per chunk, appending to the output lists. Semantics kept exactly: row order, the
`drop_duplicates` "first occurrence wins" set, `uri_to_label.setdefault` order, the
`label_names` narrowing (collected over all chunks), and the UTF-8 → cp1252 fallback,
which now restarts the whole read when a chunk fails to decode (partial output
discarded).

### Visibility (Phase 0)

`/train/status` reports `rss_mb` read live at request time (a value carried by the
last progress update would be minutes old during a head fit) and `peak_rss_mb`, the
highest reading of the run's `PeakSampler`, which every progress update from
`run_training` carries. `training.py` logs one line per phase (`phase=features
rss=1,842 MB peak=3,105 MB`), the UI shows both, and the bundle's `metrics.json` gets

```json
"resources": {"peak_rss_mb": 3105, "train_memory_budget_mb": 6963,
              "head_fit_threads": {"requested": 9, "min": 3, "max": 4}}
```

Nothing reads the block back, so `bundle_meta` needs no change. The job history
entry gets `peak_rss_mb` too — that is where "will 8 GB be enough?" gets its answer
next time.

### Phase 4 — training in a child process (in scope since 2026-09-11)

Run `run_training` in a child process started with `subprocess` (`python -m
app.train_worker`), not `multiprocessing`: the job spec goes in as JSON on stdin,
progress and the result come back as JSON lines on stdout, the child's log goes to the
inherited stderr — no pickle anywhere, the same on Windows and Linux. A cooperative
stop is a `{"stop": true}` line; a hard stop kills the child instead of abandoning a
thread that keeps training. The child writes the bundle into the registry's hidden
staging directory and the parent publishes it under its own disk lock, so the one-lock
rule of `registry.py` survives the process boundary.

What it buys: the process that serves the API never holds a training's memory — every
byte goes back to the OS when the child exits (the July "second training thrashes"
item), and an OOM kill takes the child, not the API: the job ends in an error that
says so instead of the container restarting. What it costs: a fresh interpreter per
run (~1–2 s of imports) and a cold first `/predict` of the new model (the bundle is
loaded from disk instead of handed over in memory).

| # | Task | Files | Test (written first) |
|---|---|---|---|
| 4.1 | `registry.stage` / `registry.publish` split out of `save` (behaviour-preserving) | `app/registry.py` | existing registry tests green; a staged bundle is invisible until published |
| 4.2 | `app/train_worker.py`: JSON spec in, JSON-lines progress/result out, stop line, exit codes | new module | spec round-trip; progress lines; a `TrainingInputError` becomes a user-facing error line |
| 4.3 | `JobRunner` runs trainings through the worker (`APIV3_TRAINING_ISOLATION=process`, default; `thread` keeps today's path and is what the unit suite uses) | `app/jobs.py`, `app/routes/training.py`, `app/settings.py` | a real tiny run through the worker publishes a loadable model; stop and hard stop; a killed child reads as an error naming a likely OOM |
| 4.4 | Measure: the API process' RSS after two back-to-back 30k runs, thread vs process mode | benchmark script | the numbers, pasted here |

## Tasks

Each task: failing test first, minimal change, gate run unpiped
(`./.venv/Scripts/python.exe -m pytest tests -q`, `ruff`, `mypy --config-file pyproject.toml`),
one commit. Step 0 of every phase: re-invoke `/better-coding-workflow`.

### Phase 0 — measure and make it visible (≈ 0.5 day)

| # | Task | Files | Test (written first) |
|---|---|---|---|
| 0.1 | `app/memory.py`: `rss_bytes`, `memory_limit_bytes`, `matrix_bytes`, `PeakSampler`, `ThreadBudget` | new `app/memory.py`, `tests/test_memory.py` | `rss_bytes()` > 0 and grows by ≥ 40 MB while a 64 MB array is alive; the sampler keeps that peak after the array is gone; `memory_limit_bytes` parses fake v2/v1 trees and returns `None` for `max`/absent/garbage; `ThreadBudget.for_matrix` arithmetic incl. the floor at 1 and `budget_bytes=None` → `requested` |
| 0.2 | `rss_mb` (live) + `peak_rss_mb` in `/train/status`; one log line per phase; `resources` in `metrics.json`; peak in the job history; UI row (de/en strings) | `app/jobs.py`, `app/job_history.py`, `app/training.py`, `app/static/ui/*` | the snapshot carries both fields; `metrics.json` of the tiny fixture run has `resources.peak_rss_mb > 0`; the history record carries the peak; `test_ui_i18n` stays green |
| 0.3 | `scripts/benchmark_training_memory.py` — the harness behind the tables above, using `app.memory`, every step in its own child process | new script, README pointer | Not a unit test: the script runs on `data_30k_ai.csv` and the 100k subset and prints the tables; they are pasted into this document as the **baseline** the gates below compare against |

### Phase 1 — output-identical peak cuts (≈ 1 day)

| # | Task | Files | Test (written first) |
|---|---|---|---|
| 1.1 | `train_memory_mb` setting + resolution from the cgroup limit; `GET /config` shows it | `app/settings.py`, `app/routes/system.py` (or wherever `/config` lives), `docs/configuration.md` | env `APIV3_TRAIN_MEMORY_MB=1024` → 1 GiB; unset + fake cgroup limit 8 GiB → 6.8 GiB; `0` → `None` |
| 1.2 | `ThreadBudget` wired through `deploy` → `select_c` / `cross_val_evaluate` / deploy fit; thread count in `phase_detail` | `app/deploy.py`, `app/tuning.py` | spy on `make_head` (pattern of `test_training_head_fits_use_the_capped_n_jobs`): budget = 3 × matrix bytes → every fit gets `n_jobs=1`; `thread_budget=None` → unchanged `effective_n_jobs` |
| 1.3 | Sequential word → char in `TfidfBackend.fit_transform` and `fit` | `app/vectorizers.py`, new `tests/test_vectorizers.py` | the matrix equals `hstack` of the two vectorizers fitted separately (`(a != b).nnz == 0`); `transform` unchanged |
| 1.4 | Release phase objects: `del x_tr, x_te, vec, head` at the end of every CV fold (today fold *k*'s matrices are still alive while fold *k+1* vectorizes); drop `shared_matrix` before the deploy fit. No `gc.collect()`: nothing in a fold is cyclic, so refcounting frees it on `del`, and a full collection per fold would only traverse the heap. `select_on_split`'s matrices already die with its frame on return. | `app/tuning.py`, `app/deploy.py` | Behavioural tests stay green; the effect is a benchmark number (task 0.3), stated as such in the commit |
| 1.5 | Docs: README "Memory" paragraph corrected (threads multiply the *fit* memory), `configuration.md` row, CHANGELOG `[Unreleased]` | docs | — |
| 1.6 | `auto` as an explicit value: `APIV3_N_JOBS=auto` (= `-1`, the default) and `APIV3_TRAIN_MEMORY_MB=auto` (the default); `GET /config` reports `"auto"` rather than a sentinel | `app/settings.py`, `app/responses.py`, `.env.example`, `docs/configuration.md` | env `auto` parses for both; `-1`/numbers keep working; `/config` shows `"auto"` |
| 1.7 | The threads a run is using, shown while it runs: `/train/status` carries `head_fit_threads` (the current fit's) and `threads_requested`; the status card shows "3 of 9 threads (memory budget)" | `app/deploy.py`, `app/static/ui/*` | a throttled tiny run reports `head_fit_threads=1`, `threads_requested=3` through `on_progress`; `test_ui_i18n` green |
| 1.8 | The time estimate counts the threads: `POST /datasets/analyze` predicts the deploy fit's thread count for the dataset under the current CPU and memory budgets (`planned_head_fit_threads`) and `estimated_minutes` scales the head-fit share of the anchor run (9 threads) by a measured speedup curve — not by 1/threads: 6 threads fit 3.1× faster than 1, not 6× | `app/memory.py`, `app/profiles.py`, `app/dataset_stats.py`, `app/static/ui/*` | the prediction arithmetic; fewer threads → a longer estimate, the anchor thread count reproduces the published 40.2 min |

**Gate for Phase 1** (benchmark, 100k subset): head-fit peak with `APIV3_TRAIN_MEMORY_MB`
set to *(current RSS + 3 × matrix)* stays under that budget; vectorizer peak ≤ 0.75 × the
Phase 0 baseline; a full `auto` run on `data_30k_ai.csv` before and after yields
byte-identical `metrics.json` apart from `created_at`, `training_time_seconds` and the
new `resources` block (diff the two files).

**Result (2026-09-11):**
- ✅ Output identity: the branch's `auto` run on `data_30k_ai.csv` and two runs of
  `main` give identical bundles — config, metrics minus run fields, all 9,600,000
  coefficients and intercepts of the 48 label heads, both vocabularies (200,000 terms)
  and idf (`compare_bundles.py`; two `main` runs are identical to each other too, so the
  comparison can tell).
- ✅ Head fit within budget (baseline table: 1,282 MB against a 1,492 MB budget).
- ❌ Vectorizer peak **0.86 ×** baseline at 100k (+1,097 MB against +1,276 MB in the
  same benchmark run), 0.90 × at 30k — not the 0.75 ×. The target came from the
  provisional shared-process measurement (0.70 ×), which the reuse of freed memory had
  flattered. Fitting one vocabulary after the other removes the overlap, not the peak
  of the larger fit; cutting that is Phase 2's job.

### Phase 2 — two-pass vocabulary (≈ 1–1.5 days)

| # | Task | Files | Test (written first) |
|---|---|---|---|
| 2.1 | `app/vocabulary.py`: `count_terms`, `select_terms`, `smooth_idf` | new module, `tests/test_vocabulary.py` | against `TfidfVectorizer.fit_transform` on the tiny fixture *and* a built corpus with **ties** at the `max_features` cut (equal tf, different terms) and `min_df` as a float: identical `vocabulary_`, identical `idf_` values and dtype |
| 2.2 | `fixed_vectorizer` + `transform_chunked`; `TfidfBackend.fit_transform` uses the two passes; `transform` uses chunks | `app/vocabulary.py`, `app/vectorizers.py` | training matrix and unseen-text transform identical to the reference (`(ref != x).nnz == 0`, `array_equal`) for chunk sizes 1, 7 and 10,000; word-only profile path; bundle round-trip through `model_io` unchanged (`test_model_io` green) |
| 2.3 | CHANGELOG + README ("Are the vocabulary caps cutting off signal?" gains one sentence: the caps now bound memory too) | docs | — |

**Gate for Phase 2**: benchmark vectorizer peak ≤ 0.5 × the Phase 0 baseline at 100k;
vectorization time ≤ 1.5 × baseline; the `auto`-run `metrics.json` diff from the
Phase 1 gate stays empty.

### Phase 3 — chunked loading (≈ 0.5–1 day)

| # | Task | Files | Test (written first) |
|---|---|---|---|
| 3.1 | Move the loader to `app/dataset_load.py` (behaviour-preserving; `data.py` re-exports nothing — callers import the new module) | `app/data.py`, new `app/dataset_load.py`, callers (`prepare.py`, `dataset_stats.py`, `predict_csv.py`, scripts, tests) | existing tests green; `grep -rn "data_mod.load_dataset\|data.load_dataset"` empty |
| 3.2 | `chunksize` reading with identical semantics | `app/dataset_load.py` | `LoadedData` equal for chunk sizes 1, 7, 10,000 on the tiny fixture (texts, label lists, `uri_to_label` order); gzip fixture; a cp1252 fixture whose bad byte sits in the *second* chunk |

**Gate for Phase 3**: loading the whole `data_300k.csv`, the load peak (above the RSS
before the load) ≤ 2 × what the load retains; the baseline ratio comes from task 0.3.
(The earlier "5.4×" set the whole-file peak against the text kept for the 100k subset —
two different row sets, not a ratio.)

## Expected effect

Estimates from the measured constants; every line is re-measured by the benchmark
before it is claimed.

| | today | after Phase 1 | after Phases 1–3 |
|---|---:|---:|---:|
| 30k — vectorizer peak | 330 MB | ~210 MB | ~120 MB |
| 30k — head fit, 6 threads | 645 MB | ≤ budget (unchanged when uncapped) | same |
| 100k — vectorizer peak | 1,243 MB | ~870 MB | ~400–500 MB |
| 100k — load peak (whole file) | 1,481 MB | 1,481 MB | ~400 MB |
| wlo_max `auto`, 300k usable rows, 8 GB container | dies in fold 1 | runs: base ~2.5 GB, head fits capped at ~3–4 threads instead of 9 → ~2.5× the wall-clock of an uncapped run | base ~1.5 GB → ~4–5 threads within the same 8 GB (~2× wall-clock) |

The honest summary for the owner: **Phase 1 is what turns the killed run into a
finished one**; Phases 2–3 buy back threads (time) inside the same limit and shrink
the residue; Phase 4 removes the residue entirely.

## Until Phase 1 is deployed (no code change)

`APIV3_N_JOBS=2` for the wlo_max run in the 8 GB container. The largest head fit is not
a fold's but the deploy fit, which in CV mode runs on all rows: ~0.78 GB of matrix at
300k usable rows, so 3 threads would hold 3 × 2.5 × 0.78 ≈ 5.9 GB of fit copies on a
~2.8 GB base (texts, matrix, residue) — 8.7 GB, over the limit. 2 threads: ≈ 3.9 GB on
the same base, 6.7 GB; 1 thread ≈ 4.8 GB for certainty. The deploy vectorization is the
other tight spot (~3.7 GB transient at 300k rows, word ∥ char) and no setting reaches
it. The `.wslconfig` change (VM memory) remains the owner's decision.

## Risks and how they are held

- **sklearn internals.** The per-thread factor describes `_linear_loss.py` of 1.6.1.
  A future release may drop the `hX` copy; the factor is then conservative (fewer
  threads than needed, never an OOM). The two-pass selection reproduces
  `_limit_features` line by line; if a release changes the tie-breaking, `test_vocabulary`
  fails loudly instead of the model drifting silently.
- **Memory cannot be unit-tested.** Peaks are benchmark evidence (task 0.3), pasted into
  this document with the command that produced them; the unit tests pin *behaviour*
  (identity, arithmetic, wiring).
- **Time.** Sequential vectorizers cost ~+23 % on the vectorization phase at 100k
  (provisional, see above); the two passes ~1.3× (prototype); the thread cap costs wall-clock in
  proportion to the threads it removes — only when a budget is active, and visible in
  the status line.
- **Platforms.** RSS via `/proc` and `psapi` are exercised here (Linux container,
  Windows dev box); macOS falls back to `resource` and is untested.

## Decisions (the three former open questions, settled by the owner 2026-09-11)

1. **Budget default:** `APIV3_TRAIN_MEMORY_MB=auto` — the cgroup memory limit minus
   15 % when the container has one, otherwise no cap; a number overrides it, `0`
   disables it. The `apiv3-test` container runs with **6000 MB** for testing (its env
   file, not the image). The head-fit thread count itself is automatic as well
   (`APIV3_N_JOBS=auto`), shown during the run and counted in the time estimate.
2. **`peak_rss_mb`** is a measurement, not a setting: what a run needed at most. It
   costs one number, so it goes into the bundle's `metrics.json` under `resources` *and*
   into the job history.
3. **Phase 4** is in scope: implemented and measured (tasks 4.1–4.4).
