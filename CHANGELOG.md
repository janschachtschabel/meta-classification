# Changelog

Notable changes to MetaClassify (torch-free metadata text-classification API). Dates are UTC.

## [Unreleased] — training memory (2026-09-11)

A training run on the full WLO export (426 724 records) was OOM-killed in an 8 GB
container while it fitted its first head — a phase that logs nothing. Where a run's memory
goes was measured step by step (`docs/plans/2026-09-11-training-memory.md`) and cut
without changing a single number the model produces.

### Added

- **A memory budget for training** — `APIV3_TRAIN_MEMORY_MB` (default `auto` = 85 % of
  the container's memory limit; a number in MiB; `0` = no cap). Every concurrent newton-cg
  fit holds ~2–2.5× the feature matrix in solver buffers, so before each head fit the run
  takes only as many threads as fit into the budget: slower, never a different model.
  `GET /config` reports it next to `effective_n_jobs`; wired through `docker-compose.yml`
  and the Helm chart (`config.compute.trainMemoryMb`).
- `auto` as an explicit value for `APIV3_N_JOBS`, now its default (`-1` and numbers keep
  working); `GET /config` says `"auto"` instead of a sentinel. The Docker image,
  `docker-compose.yml` and the Helm chart default to it too — they pinned `-2` / `-1`,
  which under the 60 % CPU budget gives the same thread count.
- **Memory and threads in the status.** `/train/status` gains `rss_mb` (read live),
  `peak_rss_mb` (the run's peak, sampled beside it), `head_fit_threads` and
  `threads_requested`; the status card shows both rows ("3 of 9 — held back by the memory
  budget"). Every phase logs the memory it starts from, `metrics.json` records a
  `resources` block (peak, budget, requested/min/max threads) and `/train/history` the peak.
- `POST /datasets/analyze` predicts each profile's head-fit threads under the server's CPU
  and memory budgets (`planned_head_fit_threads`, `threads_requested`) and counts them in
  `estimated_minutes` along the measured thread curve: 6 threads fit 3.2× faster than one,
  not 6×. The cost table shows the threads; the pre-flight says when a run is held back.
- `scripts/benchmark_training_memory.py` — peak and retained RSS per pipeline step, every
  step in a process of its own.
- **Training in a child process** — `APIV3_TRAINING_ISOLATION` (default `process`;
  `thread` keeps the old in-process path). A run executes in `python -m app.train_worker`
  (JSON over pipes, nothing pickled): its memory goes back to the OS when it ends instead
  of staying with the process that serves requests, a hard stop ends it at once, and an
  OOM kill fails the job with a message naming the likely cause instead of taking the API
  down — the child raises its own `oom_score_adj` so the kernel picks it, which holds
  wherever the whole container is not killed at once (on Kubernetes ≥ 1.28 with cgroup v2
  it is, unless the kubelet's `singleProcessOOMKill` is set). The child only stages the bundle; the API process publishes it under its own disk
  lock. The memory budget covers both processes: the child counts the API process'
  memory as held. `/train/status` counts the child's memory in `rss_mb`; `GET /config` reports the
  mode; wired through `docker-compose.yml` and the Helm chart
  (`config.compute.trainingIsolation`). The cost: an interpreter start per run, and the
  first `/predict` of a new model loads it from disk.
- `scripts/benchmark_training_isolation.py` — what two back-to-back trainings leave in
  the API process, thread against process mode. In the Linux container, after two 30k
  `auto` runs with both models loaded: 1,237 MB in thread mode, 321 MB in process mode;
  during a run the API process stays at 166 MB instead of 1.7–1.9 GB.

### Changed

- **The TF-IDF vocabularies are counted in two passes** (`app/vocabulary.py`) instead of
  building scikit-learn's count matrix over every n-gram before pruning — the same matrix,
  vocabulary and idf as `fit_transform`, array for array (pinned by tests; beyond 2^24
  occurrences of one term, where float32 sums stop being exact, it falls back to
  scikit-learn). At 100 000 rows the vectorizer peak fell from +1.25 GB to +0.58 GB, and
  the step got faster (66.8 s against 82.9 s). The word and char matrices are joined at
  2× the matrix instead of scipy's 3×; validation and test rows are transformed in chunks.
- **The CSV is read in chunks** (the loader moved from `data.py` to
  `app/dataset_load.py`): loading the whole `data_300k.csv` peaks at +0.46 GB instead of
  +1.48 GB, with the same result. A file that turns out not to be UTF-8 restarts the read
  as cp1252 rather than mixing encodings.
- Every error written for the operator is shown verbatim on `/train/status`, not only
  input errors; everything else stays sanitized.
- The word and character vocabularies are fitted one after the other: concurrently their
  peaks added up, for no time saving (the analyzer holds the GIL).
- A CV fold releases its matrices, vectorizer and last head before the next fold
  vectorizes, and the shared CV matrix is gone before the deploy fit.
- README "Memory" / "On 8 GB": the "all cores share one matrix, ~1× RAM" claim described
  the input matrix only — the head-fit threads multiply the fit memory (measured 11.9× the
  matrix at 6 threads).

## [Unreleased] — reused names (2026-09-11)

A check for problems with reused model and dataset names found three, each reproduced
through the API before it was fixed, plus one test-isolation leak. Creating under an
existing name was already refused with 409 everywhere; what went wrong was what still
hung on a name after its owner was gone, or before it arrived.

### Fixed

- **A share link outlived its model or dataset.** A link names its resource, not a
  version of it: deleted and re-imported under the same name, the old link handed out the
  new file to whoever held it. Deleting a model or dataset now revokes every link to it,
  matched case-insensitively.
- **A queued training could destroy a model imported under its name.** `/train` checks
  the name when a run is submitted; the import route refused a name that was training,
  not one waiting in the queue, and `registry.save` then rmtree'd the import when the
  run's turn came. The import refuses queued names (409), the run re-checks at start and
  fails before fitting with a message the operator sees, and `save` replaces a bundle only
  with `overwrite=True` (the label-pruning repair script, the one deliberate overwrite).
- **On Windows a case-variant name kept serving a deleted model.** `/predict` with
  `model_name="M"` finds the bundle `m` and caches it under `M`; deleting `m` left that
  entry, so after a retrain `M` still answered with the old weights. Deleting now drops
  every cache key that names the bundle.

### Changed

- The share-link routes (`GET /share`, `DELETE /share/{id}`, `GET /share/{id}`) moved from
  `routes/models.py` to `routes/share.py`: a link serves a model or a dataset, so managing
  links is not model management. Code moved verbatim; the OpenAPI document is
  byte-identical before and after, and `models.py` is back under 300 lines (276).

### Tests

- The suite no longer writes into the developer's real `share_links.json` (three live
  links to the test model `odd_metrics` were found there); a test pins that every file the
  app appends to at runtime points outside the repository during the suite.

## [3.2.0] — 2026-09-10

### Fixed — three layout faults in the admin UI

- **A radio was being stretched to the full width of its column.** The field rule read
  `input:not([type="checkbox"]):not([type="file"])` and set `width: 100%`, so it caught
  radios too: the mark spanned its whole grid column and pushed its own label to the far
  side of the row. That is why the options looked spread wide and *still* wrapped over
  several lines. Excluding `[type="radio"]` is the root fix — the first attempt had
  rearranged the fieldset around the symptom instead.
- **A long inline hint pushed the min-samples input out of its row.** `Min. Beispiele je
  Label` carried `(seltenere Labels fallen weg)` inside the `<label>`, which made the
  label wider than its column. The sentence moved into the help text below, where every
  other field already keeps its explanation.
- **The models table was clipped on the right.** `main` was capped at 880px, narrower
  than the table's own content — name, task, labels, two F1 columns, the evaluation
  sentence and three buttons. The cap is now `--page-max: 1200px`. Help text keeps its
  separate 72ch limit, so the wider page does not turn prose into long lines.
- Mode fieldsets became a grid (`repeat(auto-fit, minmax(14rem, 1fr))`) with each
  option's hint on its own line under its title and the mark aligned to the first line
  rather than to the gap between the two.

### Added — the upload-only import rule is asserted, not just documented

- `CLAUDE.md` and both endpoint docstrings state that models and datasets are imported by
  **file upload only, never by URL fetch**, but nothing failed if someone added a `url=`
  form field and an HTTP client to satisfy a reasonable-sounding request. It is a
  security boundary — a server that fetches a caller-supplied URL is an SSRF pivot into
  whatever the deployment can reach — so `tests/test_no_url_fetch.py` now asserts it.
- **Two independent halves, because either alone can pass while the property is broken.**
  The *source* half AST-parses every module under `app/` and fails if one imports an
  outbound HTTP client. The *contract* half reads the published OpenAPI schema and
  requires both import endpoints to take `multipart/form-data` with a `file` field, to
  expose nothing url- or uri-shaped, and not to accept `application/json` — which is how
  a `url` field usually arrives.

### Docs — Docker is a first-class install path

- The README now covers installation **with** Docker (compose and a plain `docker run`
  with the three storage variables) and **without** it, the start command for each, and
  PowerShell equivalents beside the curl examples.
- Added: how the two API-key roles differ and how to set them, what a minimal `.env`
  looks like, the three endpoints that stay reachable without a key, and the first-run
  sequence for the admin UI.

### Changed — `auto` and `best` split so every label keeps its share (plan item B3)

- **Random splitting balances rows and leaves rare labels to chance.** Measured over 20
  seeds on data_30k_ai, the rarest label (20 positives, the `min_samples` floor) lands
  anywhere from 1 to 4 positives in a validation split whose fair share is 3 — and macro
  F1 weights it exactly as heavily as the label with 2 476. `app/stratify.py` replaces
  that with multilabel iterative stratification.
- 🟢 **Measured and adopted for the CV profiles.** 26 450 rows × 48 labels, C fixed at
  32, five fold draws:

  | | random folds | stratified folds |
  |---|---|---|
  | macro F1 | 0.7075 | **0.7125** (+0.0050) |
  | spread of that number across draws | sd 0.0031 | **sd 0.0017** |
  | per-label F1 spread across draws | mean 0.0172 | **mean 0.0152** |
  | worst per-label spread | 0.1493 | **0.1227** |

  Both halves of the plan's gate met: macro F1 not worse (it improved), per-label F1
  steadier. The headline is the second row — the reported number stops wobbling with the
  draw, which is what makes two training runs comparable at all.
- **Turned on for `auto` and `best` only, and the code default stays off.** That is the
  measurement boundary, not caution about the mechanism. The benchmark isolates the fold
  splitter: `C` is fixed so the deployed model cannot move and only the thresholds — read
  off the out-of-fold probabilities — respond to the draw. On the **holdout** path
  stratifying also moves rows between train/val/test, so it changes the model itself, and
  a fair comparison there needs a test split held fixed across both arms. `fast` is the
  only holdout profile and keeps the random splitter until that measurement exists.
- Nothing starves to zero either way (0 such cases in 20 seeds × 48 labels), so this buys
  stability rather than rescuing a label that could not otherwise be learned.

### Measured — two ways to choose a decision threshold, neither adopted (plan items C2, B4)

Both proposals changed how a per-label cut is picked. Both are now measured on held-out
rows, and both stay off. The mechanisms ship as flags so the questions can be re-asked on
another target; the defaults do not move.

- 🔴 **C2 — a finer threshold grid is worse, not better.** The proposal was to replace
  `_DEFAULT_GRID`'s 19 fixed cuts with every observed score as a candidate, which finds
  each label's F1-optimal cut exactly. It does — on the split it is tuned on. Read on a
  held-out split it **loses**: **+0.0037 in-sample / −0.0115 held out** (seed 42) and
  **+0.0042 / −0.0076** (seed 7). The extra resolution buys the tuning split's noise. The
  in-sample column is the check on the sweep itself, not decoration: an exhaustive search
  cannot lose to a 19-point one on the rows both saw, so had it not won there the whole
  comparison would have been void. `scripts/benchmark_threshold_grid.py`.
- 🔴 **B4 — threshold shrinkage buys nothing once C1 is in.** Blending each label's cut
  toward the global one with weight `n_pos / (n_pos + k)` (k = 10, fixed a priori) gave
  **+0.0006 / +0.0013 / −0.0024** macro F1 across three held-out seeds against a ≥ 0.002
  gate — straddling zero. Shipped as `Profile.threshold_shrinkage_k`, off.
- **The two results explain each other, and the second one nearly went in wrong.** The
  first run of the shrinkage benchmark measured against a baseline with C1's selection
  rule *off*, because `cross_val_evaluate`'s parameter default is `False` while the
  shipped profile default is now `True`. That produced +0.0144 / +0.0035 / +0.0033 — the
  gain C1 had already banked, re-credited to B4, which is exactly the double-count this
  plan warned about for C1/C2/B4. The benchmark now reads the shipped profile so its
  baseline cannot drift from what a training actually does.
- Shrinkage is not useless, it is *redundant here*: it removes degenerate cuts (the
  spread of thresholds narrows every time, sd 0.150 → 0.126, and the cut pinned at the
  grid minimum 0.05 disappears), but C1 has already removed them from the other side by
  picking a `C` whose probabilities do not produce them. How much there is to shrink
  depends on the tuning split — out-of-fold over the whole pool gives a label ~420
  positives here, a 20 % holdout split ~107 — which is why the knob is kept.

### Changed — the decision layer moved out of `tuning.py` into `thresholds.py`

- C1 pushed `tuning.py` to 381 lines against this project's ~300 rule, and the file had
  three reasons to change: which `C` wins, where a decision cut sits, and how the result
  is scored. `app/thresholds.py` now owns the middle one — the grid search over cuts, the
  `uri -> threshold` form a bundle persists, and applying them to a probability matrix.
  291 + 119 lines, both inside the rule.
- The dependency runs one way. `thresholds.py` imports nothing of ours, the same shape as
  `label_names.py`; `tuning.py` and `deploy.py` read it and it reads neither. `macro_f1`
  went with it because it is the objective those searches maximize, and leaving it behind
  would have made the import circular.
- **Behaviour-preserving, checked rather than asserted:** every moved block is
  byte-identical to the committed original apart from one deliberate rename
  (`_tuned_score` → `tuned_score` — a name crossing a module boundary should not be
  private). Same 313 tests, same result, before and after.
- The four threshold tests moved to `tests/test_thresholds.py` alongside the module.

### Changed — C and its decision thresholds are now chosen together (plan item C1)

- **The old rule threw away candidates before they could show what they were worth.**
  `select_c` and `cross_val_evaluate` ranked every `C` at a flat 0.5 cut and tuned
  thresholds only on the winner, so a candidate whose probabilities are *ranked* well but
  *scaled* low scored near zero and was eliminated before its thresholds existed. Each
  candidate is now scored under thresholds tuned for itself, and the (C, thresholds) pair
  is picked together. The better procedure was already in this repo —
  `scripts/benchmark_field_weights.py` has always done it this way — it had simply never
  reached the pipeline.
- 🟢 **Measured on data_30k_ai (26 450 rows x 48 labels, `auto` shape) and adopted.**
  Three seeds, each drawing its own held-out split *and* its own folds; every number
  below comes from 5 290 rows that neither the C search nor the threshold tuning saw:

  | seed | macro F1 | Δ | labels/row | best_C |
  |------|----------|---|-----------|--------|
  | 42   | 0.6884 → 0.6984 | **+0.0101** | 1.525 → 1.488 (−2.4 %) | 8 → 32 |
  | 7    | 0.7243 → 0.7271 | **+0.0028** | 1.511 → 1.502 (−0.6 %) | 8 → 32 |
  | 1234 | 0.7221 → 0.7254 | **+0.0033** | 1.468 → 1.473 (+0.3 %) | 8 → 32 |

  Median **+0.0033** against the plan's ≥ 0.002 gate, positive on all three. The other
  half of the gate — the owner's condition that better decisions must not come from
  asserting more labels — passes with room: the model asserts *fewer* labels per row,
  closer to the ~1.45 the data actually carries, with macro precision up on two seeds of
  three. All three seeds moved `best_C` 8 → 32, so the effect is systematic rather than
  one lucky draw, and it costs no measurable time (233 s vs 234 s, inside this machine's
  run-to-run spread) because it adds threshold searches, not model fits.
- **Two caveats recorded rather than buried.** Micro F1 is flat (+0.0100 / −0.0003 /
  −0.0003): the gain sits in the rare labels that macro weights equally, which is what
  per-label thresholds are *for*, not an improvement everywhere. And the rule picks the
  top of the C grid every time, so widening that grid past 32 — where quality was
  measured to drop — has to be re-measured together with this flag.
- `Profile.select_c_on_tuned_thresholds` is the switch, **on** for `fast`, `auto` and
  `best`; `select_c_on_tuned_thresholds: false` in a profile turns it off. The
  gain above is the CV path (`auto`, `best`). A real-data run of the **holdout** path
  found no disagreement at all — the flat cut already picks the top C there, even with the
  grid widened down to 0.25 — so `fast` carries the flag at a measured cost of nothing
  rather than an assumed one.
- The holdout path also stopped scoring the validation split twice. `select_c` returns
  the winner's thresholds, so `deploy.select_on_split` no longer runs a second
  `predict_proba(x_va)` purely to rediscover numbers it had already computed.
- Test-first throughout, with each rule sabotaged to prove its test can fail: scoring on
  the flat cut anyway, keeping the last candidate's thresholds instead of the winner's,
  dropping either wiring, re-tuning instead of keeping, and each half of the new default.
  Writing the wiring test found a real bug — the CV path had been left inert, accepting a
  flag that `fit_evaluate_deploy` never passed it. **313 tests green**, ruff/mypy clean.

### Added — a profile can loosen the C search's convergence (measured, rejected, kept as a knob)

- `Profile.selection_tol` sets the tolerance for the SELECTION fits only. Every C
  candidate is fit to be scored once and discarded; only the deploy fit is kept, and it
  always uses scikit-learn's 1e-4. A test asserts that separation directly rather than
  trusting that both call sites were wired correctly.
- 🔴 **Measured on data_30k_ai (26 450 rows x 48 labels, `auto` shape) and NOT adopted.**
  1e-3 is harmless — same `best_C`, macro F1 moved -0.000029 against a +/-1e-4 budget —
  and saves a **median 4.4 % of the selection phase over four runs** (-2.7 / +3.4 / +5.4 /
  +11.2 %), against the plan's 15 % gate.
- The interesting part is *why*. The convergence tail is not thin: 1e-3 halves the solver
  (7.02 -> 3.69 newton-cg iterations per label; one full-row fit 26 s -> 12 s), and those
  numbers were identical in every run. The **phase** barely moves because it is mostly
  other work — k vectorization passes, the scoring, the threshold search. A first run
  reported -2.7 %, so the benchmark was repeated: on this machine a single wall-clock run
  cannot resolve an effect this size, while the solver-level measurement is exact. The
  script reports both for that reason.
- This also re-prices **A2** (warm-starting the C path, 2-3 days): it draws on the same
  pool of solver iterations that A3 just halved for 4.4 %. Recommended for dropping —
  recorded in the plan as an estimate chained across two measurements, not as a
  measurement.


### Added — the admin UI speaks German and English (plan item D6)

- **The audience reads German; the interface was entirely English.** A toggle in the top
  bar (and on the login bar, which is the screen someone without a key never gets past)
  switches between the two. The language is resolved as **stored choice → browser
  preference → German**, and the choice survives the tab.
- Every string moved out of the markup and the code into `app/static/ui/strings-de.js` /
  `strings-en.js`. Those files are strict JSON inside a one-line assignment: the browser
  loads them with a `<script>` tag (no fetch, no async boot, no half-translated first
  paint) and `tests/test_ui_i18n.py` reads them as data.
- **All of it or none of it**, checked rather than claimed: the suite fails if the two
  maps disagree on a key, if markup or code asks for a key nobody defined, if the map
  grows an entry nothing uses, or if a key is assembled from a template where no test
  can follow it. `index.html` now carries structure only — a string exists in exactly
  one place — and it is *parsed*, so untranslated text or an untranslated readable
  attribute there is caught outright. The JS side is netted rather than parsed: three
  patterns for the shapes text actually takes here (a sentence in a quoted string, a
  label between two tags, a literal handed to toast/confirm/textContent/showError).
  A net is not a proof, so each one carries a test that it still fires on a real
  example — and all three were verified against the pre-change sources first.
- Plurals go through `Intl.PluralRules` (`1 Korrektur` / `5 Korrekturen`, not `1 texts`),
  and numbers and dates through `Intl` with the **UI's** locale rather than the browser's,
  so a German UI reads `156.373` where an English one reads `156,373`.
- Known limits: the training **phase** text and the profile descriptions come from the
  server (free-form prose and `config.yaml`) and are shown as they arrive; output already
  rendered keeps the language it was rendered in until the next load, which is deliberate
  — switching language must not throw away a half-filled form or a result table.


### Added — a profile can share one vectorization across CV folds (measured, off by default)

- Cross-validation refits the vectorizer per fold so no fold's test rows shape its
  vocabulary or IDF. `refit_vectorizer_per_fold: false` on a profile trades that for one
  pass instead of k. `scripts/benchmark_shared_vectorizer.py` measures the trade.
- 🟢 On `data_30k_ai.csv` (26 450 rows × 48 labels, `auto` shape): **macro 0.7337 shared
  vs 0.7320 refit — +0.00171 — and 6.1 % of the CV phase saved.** Inside the 0.002 gate,
  but at 86 % of its budget and pointing exactly the way leakage predicts.
- **The default does not change.** The gate asks for two targets; the second (university
  subjects) is not measurable here — `data_30k*.csv` carries a single row with that
  label and the 300 k export those models came from is not on this machine. A number
  half-measured is not a number to change a default on.


### Added — corrections feed back into training (`POST /feedback`)

- **The recognition rate improves with use, or it improves only when somebody produces
  a new export.** A "Correct" button on a single-text answer records what the model said
  and what it should have said; `GET /feedback/export` hands the collection back as a
  CSV `/train` reads directly.
- Appended and never dropped, unlike the job history's 200-entry cap: this is not a log,
  it is the data the next run learns from, and the oldest correction is worth as much as
  the newest.
- Posting is **readonly**. Correcting an answer is part of classifying, and the editors
  who notice the mistakes are exactly the ones without an admin key. The export is admin:
  one correction is part of the job, every text an editor ever pasted is not.
- "None of these apply" is recordable and is left out of the export — the loader drops
  label-less rows, so including them would overstate what the file contributes.
- 🟢 The training-compatibility claim is pinned by handing the export to `load_dataset`
  itself, not by asserting a header string.


### Added — "Evaluate on…" in the model panel

- The model detail gained a section listing every dataset the model has been scored
  against — rows, **labels hit**, F1 macro and micro — and a form to start another run.
- The caveat travels with the numbers rather than living in a doc: labels hit says how
  much of the label space the dataset exercises, and the note says plainly to compare
  the same dataset across models, never two datasets.
- A run with nothing comparable shows "–" instead of a zero, matching the API.
- Verified at 360 px in dark mode: no overflow, the table scrolls in its own wrapper,
  every target ≥ 24×24, no CSP or console errors.


### Added — evaluate an existing model on a dataset (`POST /models/{name}/evaluate`)

- "Model B beats model A" is only a statement if both were measured on the same rows.
  Until now that meant a script driving a running server (`scripts/eval_holdout.py`),
  which nothing recorded and nobody could repeat. The result is now **appended to the
  bundle's `evaluations`** — beside the training metrics, never over them.
- Truth is binarized over the **model's own classes**, not the dataset's: the two are
  different widths, and lining up column 0 of one with column 0 of the other measures
  nothing. Labels the model never learned are reported; rows carrying only such labels
  are excluded and counted, because scoring them as failures blames the model for a
  label it was never given and dropping them silently flatters it.
- 🟢 `labels_covered` came out of the live check: `faecher_300k_auto` on an 8-row probe
  scored **f1_macro 0.068 beside f1_micro 0.941**. Both correct — macro averages over
  all 59 classes and 55 had no examples. The coverage number turns that headline from
  "this model is terrible" into "this data exercises 4 of its labels".
- When nothing comparable is left, `metrics` is `null` rather than 0.0: an F1 of zero
  reads as "terrible here" instead of "these two vocabularies do not meet".
- Runs on the same single worker as training, so it queues the same way; the history
  tells them apart with `kind`.


### Removed — the browser no longer holds the training queue

- Selecting several label fields posted the first run and kept the rest in a JavaScript
  array, shifting one out on each poll tick. Every run now goes to the server at once
  and the server holds the order — the tab can close, the laptop can sleep.
- The status card shows what is waiting from the server's own answer
  (`GET /train/status` → `queued`), not from page state.
- The name preview no longer says "keep this tab open", because that is no longer true.


### Changed — a second training is queued, not refused

- `POST /train` while a run is going answers **202 with a `queue_position`** instead of
  409. The server starts the next one when the current finishes, so "train five label
  fields" no longer needs a browser tab kept open to shepherd it — the queue used to
  live in the page, and closing it lost every run that had not started.
- Bounded to 10 waiting runs. A name already running or queued is still a 409: that run
  could only fail, since `/train` refuses an existing model, so it is refused while it
  is still a request.
- **Stopping clears the queue** — "stop" means "end this", not "skip to the next one".
  The browser-side queue already behaved this way; the server now keeps that promise.
- `GET /train/status` lists what is waiting in `queued`.
- `start()` keeps its old strict meaning (start now or refuse) and the guard the
  existing concurrency tests pin; `submit()` is the new entry point that queues. The
  dispatch is the finishing thread's last act, in a `finally` so a queue never strands
  behind a run that failed in an unanticipated way.


### Added — finished runs are written down (`GET /train/history`)

- Comparing two runs meant opening two bundles and reading their `metrics.json`, and a
  run that **failed** left nothing to open at all — its reason lived only in whichever
  browser tab happened to be watching. Every finished run now leaves a record: the
  request it was started with, how it ended, its duration, and the headline scores.
- 🟢 Verified across a real process restart: `/train/status` resets to `idle`, the
  history entry and its metrics do not.
- The **resolved profile** is recorded even though it is not part of the request
  dictionary the pipeline uses — it is the dimension two runs differ on most, and a
  comparison without it compares nothing.
- Not the full metrics: `per_label_f1` is one entry per label. The history exists to
  compare runs, which needs the numbers a comparison is made on; the bundle keeps the
  rest. Bounded to the newest 200, rewritten atomically, and a damaged line costs that
  line — never the endpoint, and never a run, which writes here only after the work is
  already saved.


### Added — a pre-flight check in the Training tab

- **Check before training** reads the dataset once and answers the two questions you
  otherwise only get after the run: how many labels *your* threshold keeps, and what the
  run costs on the selected profile (per model, when several label fields are planned).
- On the fixture this immediately catches the real failure: the form's default threshold
  of 20 "keeps 0 of 3 labels" — a run that would have stopped with "not enough data".
  A button puts the heuristic's value into the field.
- On demand, not on every change of the label field: it parses the whole CSV, and a
  300 k export costs half a minute each time. The button says what it costs.
- `app.js` split three ways along the seam the new view exposed: the shell (126),
  `training.js` — the form, the pickers, the browser-side queue (227) — and
  `train-status.js` — the poll loop, the status card, the topbar chip (112). They change
  for different reasons: what a run is *configured* with, versus what it *reports*.


### Added — a dataset inspector in the Datasets tab

- Clicking a dataset name shows its columns and first rows, and offers **Analyze**:
  how many labels survive each threshold, what a run costs per profile, and a warning
  when labels with fewer than ten examples are in the set.
- The recommended threshold is marked three ways — bold, a background, and the word
  "recommended" in its own column — never by colour alone.
- The cost table says "under a minute" where the model rounds to 0.0: at 36 rows a model
  anchored at 156 373 has nothing precise to say, and "0.0 min" would imply it does.
- Analysis runs on demand, not on open: it parses every row, which is a 195 MB job on a
  real export.


### Added — analyze says what a run will cost, before you start it

- `POST /datasets/analyze` now answers with `recommended_min_samples_per_label` and
  `estimated_minutes` per profile. Both are the two numbers that decide whether to start
  a training run at all, and both are computed server-side so nothing re-derives them.
- The cost model (`app/profiles.estimated_minutes`) is anchored on the one full-scale
  measurement this project has — `faecher_300k_auto`, 156 373 rows, 40.2 min — and
  scales linearly, which is what `benchmark_row_scaling.py` measured (exponent 1.00).
  🟢 It reproduces **all six** figures published in the README's per-profile table
  (4.0 / 40.2 / 72.4 min at 156 k; 15.4 min / 2.6 h / 4.6 h at 600 k). A profile with no
  measured factor gets `None`, not `auto`'s number.
- The threshold table now always contains the recommended value. On a small dataset the
  heuristic answers 2, which none of the fixed buckets covered — leaving the
  recommendation unreadable as the sentence it exists for: "keeps N of M labels".


### Added — "Why?" on a single-text answer

- `/predict/explain` has existed since 3.1; nothing in the UI reached it. A **Why?**
  button on each model's answer now shows the words that carried it, per predicted
  label, as chips with their leave-one-out impact — plus every label's score, collapsed.
- **The bars are scaled per label, not across them.** Measured on the real
  `faecher_300k_auto`: a confident answer moves by 0.002 for its strongest word (0.9997
  confidence barely falls for anything), while a contested label swings by 0.55. One
  shared scale would have flattened the first case to an empty row. What the bars carry
  is the order; the numbers carry the amount, and the panel says so.
- A word that argues *against* a label is marked three ways — a `−` prefix, a dashed
  border and a muted bar — because colour is not a signal on its own.
- A single-word text gets an honest note instead of an empty block: leaving a word out
  needs at least two. The label table is still there, which is the useful part then.
- An explain call that fails leaves the answer it belongs to standing, reports inside
  that card with `role="alert"`, and a retry replaces it.


### Added — the Query tab classifies lists and files, not just one text

- Three modes: **one text** (as before), **many texts** (one per line, sent in batches
  of 500, answered as a table you can download as CSV), and **a CSV file** (uploaded to
  `/predict/csv`, answered as a CSV download plus a receipt).
- A text the model asserts nothing for is listed with an empty label in both bulk modes
  — "which ones did it refuse" is part of the answer, not something to infer from a gap.
- A `<fieldset>` of radios, so grouping, the group name, arrow-key navigation and the
  single tab stop come from the browser. The reliability-signal checkboxes hide in CSV
  mode: that answer has no column for them, and a control that does nothing is a lie.
- Verified in the running UI: keyboard-only mode switching, 360 px without horizontal
  scrolling, every target ≥ 24×24 (the file input was 23 px — fixed for every upload
  field), dark mode via the tokens, the outcome announced on the `aria-live` status,
  no CSP or console errors.
- `query.js` split out of `app.js` (389 / 236).


### Added — classify a whole CSV in one call (`POST /predict/csv`)

- Upload a CSV, get a CSV back. The daily editorial job is "classify these 500 new
  items", not one text. `/predict` could always do it, but only if the caller assembled
  each row's text — and **how a text is assembled is part of what the model was fit
  on**, so that is the one thing not to leave to the caller: the text columns and their
  repetition weights are read from the bundle.
- `row,uri,label,confidence,above_threshold`, one line per predicted label, `row` being
  the 0-based input row so the answers join back onto the original file. **A row the
  model asserts nothing for still gets a line** — "which items did it refuse" has to be
  readable off the result, and a vanished row is indistinguishable from one never sent.
- Neither side is materialised. 🟢 Measured: 50 000 rows against the 59-label
  `faecher_300k_auto` → 4.1 MB of CSV in 54.6 s at a **2.2 MB peak heap** on a 6.5 MB
  input; `transfer-encoding: chunked`, no `content-length`.
- The text assembly now lives in one function (`data.combine_text_columns`) shared with
  training. Two loops that merely looked alike would have drifted into exactly the
  train/serve skew the model card warns about — silently, with plausible numbers.


### Changed — exports and dataset uploads no longer pass through memory

- Exporting a model built the whole archive in RAM. Measured on the real 51 MB
  `faecher_300k_auto`: **142 MB of peak Python heap, 2.78× the bundle** — every member,
  the zip built beside them, and the copy `getvalue()` makes. It now streams member by
  member into a staging file which the response sends and then deletes: **2.2 MB peak,
  0.04×**. For a 180 MB bundle on a small vServer that is the difference between an
  export and an OOM.
  - The trade, stated plainly: compression happens under the disk lock now, where before
    only the read did — 3.1 s for that bundle. The members are read one at a time from
    files that must still exist, and holding open handles outside the lock would make
    `delete` fail outright on Windows.
  - Staging goes beside the bundles, not into `/tmp`: on a container that is often tmpfs,
    i.e. RAM. It uses the hidden `.*.tmp` name the startup sweep already cleans.
- Dataset uploads (capped at 200 MB, and WLO exports really are 126–195 MB gzipped) were
  assembled in memory — the chunks *and* the joined copy, twice the file, before a byte
  reached disk. They now spool straight into `<name>.part` and are renamed into place.
  The rename is what makes a dataset exist, so a failure before it leaves nothing that
  can be listed, inspected or trained on.
- **Model import still buffers** (172 MB peak, 3.37×). `unpack` validates the archive as
  bytes *before* anything touches the filesystem, and that order is the security property
  — streaming it means extracting to staging first, which is a separate change.


### Added — model detail view

- Clicking a model's name opens a panel with everything the bundle records about
  itself: dataset, text columns with their weights, profile, rows, the searched `C`
  grid, evaluation, duration, created — and **every label with its F1, its row count
  and the threshold serving actually applies**, sortable, weakest first.
- When the winning `C` sits at either end of the searched grid, the panel says so:
  the optimum may lie outside it, and the fix is a wider grid, not more folds. The
  first real model checked showed exactly that (`32.0` out of `[2.0, 8.0, 32.0]`).
- A native `<dialog>`: the focus trap, Esc, and returning focus to the row are the
  platform's, not hand-written. Verified at 360 px with no horizontal scrolling and
  every target ≥ 24×24.
- `manage.js` split into `share.js` (used by both tabs), `models.js`, `datasets.js`
  and `model-detail.js`.


### Fixed — smaller things found while measuring the above

- **Export packs an allowlist**, not "everything except two names". `update_info`
  stages a `metrics.json.tmp` beside the file it replaces, so a crash in that window
  left a member export would pack and import would then refuse: an archive this API
  produces but does not accept.
- **A bundle with a damaged JSON document can be exported again.** Generating the card
  made export the one operation that *parses* a bundle, so the bundle most in need of
  being exported — a damaged one, to inspect elsewhere — was the one export refused
  with a 500. The card degrades; the members travel byte for byte.
- **The share-link overview refreshes on every path.** It was rendered after the model
  table, so the two paths that skip the table (no models left, or the list request
  failing) left the previous overview on screen — offering links that are still live.


### Fixed — the share-link overview no longer shows dead links as live

- Expiry was enforced in two of the three readers: the store purges at startup and
  `resolve` refuses at use, but `GET /share` had **no check at all** — so a link that
  expired while the server was running was still listed as outstanding by the one
  screen an operator revokes from. There is now a single definition of "live" that
  all three share; the bug was the third copy that was never written.
- A share-links file that is valid JSON of the wrong shape (an array, say) resets
  with a warning like an unreadable one, instead of raising on the first share route.


### Fixed — a bundle's own metadata can no longer take down a download

- `metrics.json` travels **inside** importable bundles, so every value in it is
  foreign input. The serving path always read it that way; the two *reporting*
  consumers — the model card and `GET /models/{name}/labels` — formatted its values
  directly. Eleven of twelve reads had no type guard (measured), so one wrong type
  crashed the export that packs the card, and with it `GET /share/{id}` — **the one
  route with no API key**, which answered 500 to an anonymous caller.
- The discipline now has one home: `app/bundle_meta.py` (`as_mapping`, `as_names`,
  `as_number`, `as_count`, plus `per_label_f1`, moved out of `model_io`). Reporting
  data may legitimately be absent — bundles predating a field have none either — so
  an unusable value degrades to "—" instead of raising.
- Table cells in the card escape `|` and newlines: an author-supplied name could
  otherwise break the table it sits in.
- An import whose `config.json` has a `classes` that is not a list is refused with
  **400** instead of 500: the container-label warning read that field *outside* the
  block that maps shape errors, so it escaped as a bare `TypeError`.
- A damaged archive is a 400 again, not a 500. Reading members moved into
  `model_archive.unpack()` during the extraction, out of the handler that mapped
  `BadZipFile`. Corruption modes measured rather than assumed: a mangled member name
  raises `BadZipFile`, a **flipped data byte — the likeliest damage — `zlib.error`**,
  a truncated stream `ValueError`/`EOFError`.

### Added — per-label diagnostics (`GET /models/{name}/labels`)

- A model's headline F1 says how good it is on average; this says **where it is
  weak**, which is what decides whether one answer deserves a second look. Each
  label with its F1, its support and its threshold, **weakest first**.
- `per_label_support` (rows carrying the label) is recorded at training time from
  the full label matrix — **not** from `compute_metrics`, which the plan proposed:
  under a holdout split that function only sees the test share, so the number would
  report ~15 % of the examples while reading as the real count. It is what makes a
  score interpretable: 0.13 on 25 rows is a different statement from 0.13 on 5,000.
- `threshold` is `null` for binary/multiclass, where serving decides by argmax and
  never reads one — reporting the global 0.5 there would describe a rule the model
  does not apply. 🟢 Confirmed on the real store: 59-label multilabel model reports
  its tuned cuts, a 70-label multiclass model reports none.
- Bundles trained before this release report `support: null` and keep working; the
  model card's weakest-label table gains the row count where it is available.


### Added — share links can be reviewed and withdrawn

- `GET /share` lists every live link (id, kind, name, created, expires) and
  `DELETE /share/{id}` revokes one; both admin-only, because the id **is** the
  capability. Handing a link out was always possible — seeing what was outstanding,
  or taking it back before its expiry, was not: the only way was editing
  `share_links.json` on the volume.
- Links now record `created_at`, so an overview can say how old one is. Entries
  written before this release list with `created_at: null` rather than breaking the
  view — verified against the real store (33 entries, 29 of them without the field).
- Admin UI: an "Active share links" table on the Models and Datasets tabs, each row
  with Copy link and Revoke.
- **Not** implemented, deliberately: download counting and `max_downloads`. Counting
  means a JSON write on every hit of the *public, unauthenticated* download route,
  for information the access log already carries; the gap that was asked for is
  revocation.


### Added — exported bundles explain and verify themselves

- Every export now carries a generated **`README.md` model card**: what the model
  classifies (including the derived label vocabulary), the author's own statements,
  how it was trained (dataset, columns, weights, profile, `C` and the searched grid,
  rows, features, duration), how well it scores, and **its ten weakest labels** — the
  ones where a high confidence is worth least. Where a model was trained with
  `text_column_weights`, the card repeats the train/serve consistency warning, since
  a recipient cannot know it otherwise.
- And a **`manifest.json`** with a SHA-256 for every other member. Import verifies it
  and refuses an archive whose member was altered or arrived truncated; previously
  only skops choking on it would have caught that. An archive **without** a manifest
  still imports, so bundles shared before this release keep working.
- 🟢 Verified on the real 44 MB `faecher_300k_auto` bundle: export 2.0 s, six members
  hashed, card generated with correct UTF-8 (2638 bytes for 2625 characters).
- **Deviation from the plan, deliberate:** the manifest is built at *export* and
  checked at *import*, not written at save and checked at load. `metrics.json` is
  mutable by design (`PUT /models/{name}/info`) and `config.json` is rewritten by the
  label-repair scripts, so a stored manifest would be invalidated by every legitimate
  edit — and a load-time check would then reject a perfectly good bundle. The threat
  worth guarding is the 50-180 MB transfer between servers, which is exactly this
  boundary.
- **Compatibility, one direction only:** an instance running 3.1.0 or earlier will
  *reject* an archive exported by this version, because its member allowlist does not
  know `manifest.json` and `README.md`. Upgrade the receiving side first.


### Added — model documentation that travels with the bundle

- `PUT /models/{name}/info` stores `author`, `description` (purpose **and** limits),
  `data_source` and `license` in the bundle; `/train` accepts the same block up front.
  These are the facts the pipeline cannot measure, and they matter at exactly one
  moment: handing a model to a third party, where `dataset: "data_300k.csv"` says
  nothing about where that file came from. They live in `metrics.json`, so they travel
  inside the exported archive, and they are editable at any time — documentation is
  presentation-only, so correcting an author name costs no retrain.
- `GET /models/{name}` reports **`label_vocabulary`**, derived from the class URIs
  rather than typed in: a vocabulary is the direct parent of every concept id, which
  is what tells one vocabulary from the common ancestor of two. 🟢 Verified against
  all 14 bundles of the local model store (discipline, educationalContext,
  hochschulfaechersystematik); a mixed label set correctly reports none.
- Admin UI: an **Info** button per model opens the form and shows the derived
  vocabulary read-only. Free text is escaped on render — `metrics.json` arrives
  inside importable archives.

## [3.1.0] — 2026-09-09

First release since the torch-free rewrite. It carries the whole July development
wave and the 2026-09-08 audit remediation; the dated sections below document them
in the order the work happened.

**Breaking.** Strict semver would call this a major bump — it ships as 3.1.0 by
owner decision, so read these three before upgrading:

- **Model bundles are format 2.** A bundle trained before it keeps its vocabulary
  inside the skops container, no longer loads, and has to be retrained. The API
  says so explicitly instead of answering 404.
- **`POST /train` answers 202**, not 200: it accepts a job and returns a status
  URL. A client asserting `== 200` breaks; one asserting `< 300` does not.
- **`POST /datasets/{name}/validate` takes one JSON object** like
  `/datasets/analyze`. The former raw-array body plus query parameters are gone.

## audit remediation, phase 0 (2026-09-08)

### Added — every warmed model stays resident (`APIV3_WARMUP_MODELS`)

- Listing models for warmup states that each should answer without a cold skops
  load. The LRU was sized independently from `APIV3_MAX_MODELS_IN_MEMORY`
  (default 2), so warming three models evicted one during startup and its first
  request paid the load anyway — documented as a rule to follow ("keep the count
  ≤ the LRU size") rather than as the defect it was.
- The cache is now sized to `max(cap, len(warmup list))`; the cap keeps its
  meaning as the RAM ceiling for every model that is *not* warmed. The raise is
  logged at startup and `GET /config` reports `effective_max_models_in_memory`.
- The setting is wired through `docker-compose.yml` and the Helm chart
  (`config.limits.warmupModels`). It had existed only in `.env.example`, so the
  containerized deployment it exists for could not configure it.
- 🟢 Measured with the cap deliberately at 2 and three real bundles warmed:
  3 of 3 resident (2 before), every `/predict` ≈ 0.09 s.

Test-first; **196 tests green**, ruff/mypy clean, `pip-audit` clean. Acts on
`docs/audits/2026-09-08-audit.md`; the remaining phases are in
`docs/plans/2026-09-08-ergaenzungsplan.md`.

### Fixed — every bar in the admin UI rendered full width

- The UI runs under `default-src 'self'` without `'unsafe-inline'`, which blocks a
  style **attribute** exactly as it blocks an inline script. All three bars injected
  `style="width:N%"` through `innerHTML`, so the browser dropped it and each bar
  inherited its container's full width. 🟢 Measured on `/ui/`: five ranked
  predictions carrying 100/0/0/0/0 % all drew 171 of 171 px — a 0.004 label looked
  exactly like a 1.000 one, and the training progress bar sat at 100 % from the first
  second.
- Bars now carry `data-width` and `applyBarWidths()` sets `el.style.width` after each
  render (the CSSOM is not governed by `style-src`). 🟢 After: 0 px at 0 %, 497 px at
  100 %, progress 301 of 814 px at 37 %, chip 24 of 64 px, and **zero CSP console
  errors** where a single page visit had logged 80+.
- `test_ui_sources_contain_no_inline_style_attributes` pins it at the source level,
  next to the existing inline-handler guard.

### Fixed — bundles that cannot be served were offered as models

- A `<name>.prebackup` copy (kept by `scripts/prune_bundle_labels.py` before it
  repairs a bundle) is a valid bundle, so `list()` offered it — predicting with one
  serves precisely the pre-repair weights. The registry skips the suffix and the
  script imports the constant, so the two cannot drift.
- Pre-format-2 bundles fail to load (`/predict` → 422) but the Models tab showed
  their metrics like any other row. They stay listed and downloadable, now dimmed
  with a "needs retraining" badge; a 422 in the Query tab explains what to do.

### Fixed — security hardening (audit S1, S3, S4, S5)

- **Dataset routes are confined to datasets.** Only the listing filtered on the CSV
  suffixes, so inspect/analyze/validate/export/share and **delete** reached any file
  a safe name could name — including `data/label_names.json`, the display-name
  sidecar every later training reads. All six routes resolve through one
  `_dataset_path()` helper.
- `safe_name` caps a name at 100 characters (an over-long name raised inside a write
  and surfaced as an opaque 500).
- Startup **refuses** `APIV3_AUTH_ENABLED=true` without `APIV3_API_KEY_ADMIN` (no
  request could ever reach the admin role) and warns when both keys are identical.
- The `.env` path is anchored to the app directory like every other default;
  relative, it was silently dropped when uvicorn started elsewhere.

### Changed — `POST /train` returns **202 Accepted** (breaking for `== 200` checks)

It queues a job and returns a status URL; the training has not happened yet. Also:
`GET /` redirects to `/ui/` (or `/docs`), `/favicon.ico` serves an inline SVG, both
were 404. `stop(hard=true)` is a no-op while idle — it used to discard the finished
run's metrics. Shutdown asks a running training to stop so it can end at its next
checkpoint inside the termination grace period, which the Helm chart already
promised while nothing performed it.

### Added — weekly `pip-audit` job

The dependency tree is hash-pinned, so a CVE published against an already-pinned
version is invisible until someone pushes. The scan now also runs Mondays.

## prediction reliability, measured tuning, sizing to 600k (2026-07-25)

Test-first; 170 tests green, ruff/mypy clean.

### Fixed — the bundle save was quadratic in the vocabulary size (**format 2, breaking**)
- Writing `vectorizer.skops` for a 200 000-term vocabulary took **45–85 minutes** and
  148 MB — longer than training the model it was saving. Profiled: skops walks a dict
  entry by entry and is **super-quadratic in the entry count** (4k terms → 0.7 s,
  16k → 8.3 s, 64k → 296 s; exponent rising from 1.8 to 2.6). The 48 MB float head, one
  single array, wrote in a second — so it was never about size.
- The vocabularies now travel as a plain `vocabulary.json` member and the estimators go
  into skops without them. Measured at the real 200 000 terms:

  | | before | after |
  |---|---:|---:|
  | Save | 45–85 min | **0.3 s** |
  | Load | 29.1 s | **0.2 s** |
  | Vectorizer data | 148 MB | **3.0 MB** |

  Also fixes the slow first request after a model switch (a 29 s cold load becomes
  0.2 s), and shrinks every model export/import accordingly.
- Validated at load like any other bundle content: `vocabulary.json` arrives inside
  importable archives, so a wrong length (which would silently build a feature matrix
  of the wrong width) or duplicate terms are rejected with `UnsafeModelError`.
- **Breaking: `FORMAT_VERSION` 1 → 2. Existing bundles must be retrained** — a format-1
  bundle now fails to load with an explicit message rather than a misleading 404.

### Added — datasets may be gzipped (`.csv.gz`)

- Upload, listing, inspection, download and training all accept `.csv.gz` alongside `.csv`.
  Motivation: the WLO full exports are **126–195 MB compressed against ~1.4 GB plain**, and
  pandas reads the compressed form natively — so only the surrounding API stood in the way.
- Reading already worked; three places did not. `GET /datasets` globbed `*.csv`, which never
  matches `*.csv.gz`; the upload demanded a `.csv` suffix; and the download announced
  `text/csv` for gzip bytes, which lets a browser silently decompress and save something that
  no longer opens under its `.gz` name.

### Fixed — `count_rows` was off by up to 5x on real data

- It counted physical lines, so every newline inside a quoted description started a new
  "row". 🟢 Measured: `data_300k.csv` reported **1,343,683 rows for 340,630 records (3.94x)**
  and the combined WLO export **2,141,123 for 426,724 (5.02x)**. Documented as a harmless
  approximation, but at that factor it is simply a wrong number in every dataset listing.
- Now tracks quote parity per line — one cheap pass, exact for well-formed CSV (an escaped
  `""` contributes two quotes and leaves the parity untouched). A full `csv.reader` pass
  would also be exact but parses every field for a number nobody trains on.
- Independent of the gzip work: plain CSVs were equally affected. Gzip merely made it
  spectacular, since counting newlines in COMPRESSED bytes is meaningless (1,030,307 for the
  same 426,724 records).

### Changed — `best` is now `auto` + more folds, nothing else (and CV10 is out)

Both of `best`'s former distinctions were measured and only one survived.

- **The `C` grid loses its wide reach.** `best` searched `[0.5, 2, 8, 32, 128]` on my
  hypothesis that four consecutive runs selecting the grid *maximum* (16 → 32 → 128) meant
  the optimum lay beyond it. 🟢 `scripts/benchmark_c_range.py` (new; holdout split, one fit
  per candidate, zero convergence warnings) refutes it on **both** targets — macro F1 peaks
  at `C=32` and then declines:

  | C | 8 | **32** | 128 | 512 | 2048 |
  |---|---:|---:|---:|---:|---:|
  | school (156k rows, 59 labels) | 0.7504 | **0.7531** | 0.7511 | 0.7493 | 0.7479 |
  | university (30k rows, 118 labels) | 0.7803 | **0.7814** | 0.7812 | 0.7785 | 0.7785 |

  Micro F1 declines monotonically too (school 0.8603 → 0.8517) while fit time rises ~50 %.
  The repeated edge picks were **ties on a flat plateau**: at a fixed 5-fold CV, `C=32` and
  `C=128` scored 0.8135 vs 0.8130. `best` now shares `auto`'s `[2, 8, 32]`, and
  `test_no_c_grid_reaches_past_the_measured_useful_range` stops the range being re-widened.
- **The fold count is the real lever, and 5 is the knee.** 🟢 Isolated on the university
  target at a fixed `C=32`: 3 folds → 0.7992, **5 folds → 0.8135 (+0.0143)**, 10 folds →
  0.8166 (**+0.0031 for double the run time**, 16.7 vs 9.1 min). So the +0.0138 previously
  credited to the wider `C` grid was the fold count all along. **No shipped profile uses
  `cv_folds: 10`** — still available per request, but not worth 2× for three thousandths.
- Net: `best` drops from 21 to **13 head fits — 38 % cheaper at slightly better quality**
  (0.8135 vs 0.8130). At 156k rows ~1.2 h instead of ~2.0 h.
- Two of my own tests asserted the refuted design (`best` must probe past both optima; the
  candidate count must strictly increase along the ladder). They were **replaced by the
  measurement**, not relaxed to pass: the new pair pins the upper bound at 32 and forbids a
  costlier rung from searching more coarsely than a cheaper one.

### Fixed — a namespace was being trained as a class

- A label value ending in `/` names a *container*, not a concept. 🟢 Measured on
  `data_300k.csv`: the bare vocabulary root `…/vocabs/discipline/` was attached to **522
  rows** and trained as an ordinary class scoring **F1 0.4096**, so it diluted macro F1 and
  `/predict` could answer with a label that carries no meaning. The higher-education root
  did the same on 290 rows (F1 0.6888).
- `min_samples_per_label` structurally cannot catch this — 522 rows clears any sane
  threshold — so `data.split_labels` now drops container values, and a row left with no
  label is dropped like any other unlabelled row. Deliberately narrow: a genuine broader
  concept has an id (`…/discipline/120`) and is kept, because the label hierarchy is real
  signal (the higher-education vocabulary averages 2.25 levels per row). The 11 other
  malformed URIs in that file (label *text* where the concept id belongs, e.g.
  `…/discipline/Physik`) were already caught by `min_samples_per_label` at 2–10 rows.
- **Added `scripts/prune_bundle_labels.py`** — removes the class from an already-trained
  bundle, again **without retraining**. In multilabel mode `predict_proba` stacks one column
  per entry in `estimators_`, so dropping an entry drops its column; the script *proves*
  this by comparing probabilities for the kept labels before and after and refusing to write
  if anything moved. `f1_macro` is recomputed exactly (it is the unweighted mean of the
  per-label scores); `f1_micro` / `precision_macro` / `recall_macro` cannot be derived
  without the original predictions, so they are moved to `metrics_before_pruning` rather
  than left looking current. Applied: `faecher_300k_auto` 60 → 59 classes
  (macro 0.7310 → **0.7365**), `hochschulfaecher_300k_auto` 119 → 118 (0.7982 → **0.7992**).
- **Import stays covered.** Training can no longer produce such a class, but `POST
  /models/import` installs a bundle carrying its own `classes`, so an older bundle would
  reintroduce it. A loader cannot repair it — the class list is positionally tied to the
  head's estimators, and dropping one without the other would shift every probability onto
  the wrong label — so `model_io` logs a warning naming the offending labels and the repair
  command instead of failing an otherwise usable bundle.

### Fixed — display names were silently attributed to the WRONG label

- A label column and its `_DISPLAYNAME` twin are both separated by `label_separator`,
  but only URIs are guaranteed free of that character. A name containing a comma
  (`"Rechts-, Wirtschafts- und Sozialwissenschaften"`) split into two entries, so the
  positional `zip` shifted every later name onto the wrong URI — and `setdefault` then
  froze that wrong name permanently.
- 🟢 Measured on `data_300k.csv` (340 630 rows): **6.09 % of rows misaligned**, and in the
  trained bundles **34 of 119** higher-education labels (28.6 %) and 2 of 60 school labels
  carried the name of a *different* subject. Before the fix, one prediction on an
  engineering text displayed **"Physik" three times** — for Physik/Astronomie, for
  Ingenieurwesen allgemein, and for Physik.
- `data._pair_names` now pairs names only when the counts provably line up. When they do
  not, `_rejoin_split_names` reconstructs the split using the URI count as the arity
  constraint plus German orthography (a lowercase start continues, `"Rechts-"` is half a
  compound, an unclosed `(` must close). 🟢 That recovers 29.6 % of damaged rows and
  agreed with independently-clean rows **75/75 times**. Rows that still do not reconcile
  contribute **nothing** — a missing name falls back to the URI, which is honest, whereas
  a wrong subject name is not.
- **Added `scripts/fetch_vocab_labels.py`** — the complete source. A comma-corrupted
  export cannot be fully repaired from itself (data-derived names cover only 59 % of
  labels, 70 % with reconstruction), so this downloads the SKOS vocabularies once and
  writes `data/label_names.json`. Training reads that sidecar if present and lets it
  override CSV-derived names. Build-time only: `app/` still never fetches a URL.
  🟢 Cross-check against the 191 labels known from both sources: **identical, 0 conflicts**;
  coverage of the labels in use rose to **100 %**.
- **Added `scripts/patch_bundle_labels.py`** — repairs an existing bundle in place, since
  `uri_to_label` is presentation-only JSON in `config.json`. **No retraining needed**;
  it asserts `classes`, thresholds and both skops members stay untouched, and keeps a
  `config.json.bak`. Applied to `faecher_300k_auto` (2 names) and
  `hochschulfaecher_300k_auto` (34), with confidences verified bit-identical afterwards.
- Surfaced two data defects worth fixing upstream, not worked around here: 522 rows are
  tagged with the bare vocabulary URI `…/vocabs/discipline/` (no concept — it trained as a
  junk label scoring F1 0.410), and 11 more put the label *text* where the concept id
  belongs (`…/discipline/Physik`). The real fix is the export: quoting the fields, or a
  separator that cannot occur inside a name.

### Optimization summary — what was measured and what it bought

Every line below is a measurement on `data_30k_ai.csv` (48 subject labels, identical
rows/split/threshold procedure), not an expectation. Scripts in `scripts/benchmark_*.py`.

| Change | Quality | Cost |
|--------|---------|------|
| **Field weights** title+keyword 2× (now the default) | **+0.0145 macro** | none (nnz unchanged) |
| **char n-grams (3,5) → (5,5)** | **+0.0024 macro** (0.7084 vs 0.7060) | **−62 % matrix**, 3× faster fits |
| **C grid 8 → 3 candidates** | **±0.0000** (identical pick) | **−62 % compute** |
| **Label matrix int64 → int8** | none | **−87 % target RAM** (1.34 GB → 168 MB @600k×300) |
| `fast`: drop the 50 k word cap | **+0.0123 macro** | negligible |
| Vocabulary caps 80k/120k → doubled | −0.0005 (nothing) | +88 % head RAM — **not adopted** |
| Word-only instead of word+char | −0.0072 clean, but **3.6× worse under typos** | −90 % matrix — **not adopted** |
| Linear SVM head | tie within noise | needs calibration: 6× RAM, 9× latency — **not adopted** |
| LogReg ⊕ SVM ensemble | *worse* than the better member | 2× everything — **not adopted** |
| Static embeddings ⊕ TF-IDF | −0.0014 to −0.0076 | 2.2× fit time + a dependency — **not adopted** |

Net effect on the default path: **`auto` at 100k rows went from ~132 min to ~19 min**
with slightly better quality, and the profile restructure below cut it by another ~46 %
(7 head fits instead of 13). 🟢 Anchor measurement: `auto` over **156 373 rows × 60
labels in 40.2 min**, bundle save 3 s — which puts 600k rows at ~2.6 h, not the ~1 h an
earlier doubly-derived estimate claimed (see README sizing).

### Changed — the profile set is a cost ladder: `fast` < `auto` < `best` (**breaking**)
- `Profile` gained `cv_folds`, so a profile carries its own evaluation mode. Resolution
  is most-specific-first: **request > profile > config** (`split.cv_folds` in
  `config.yaml` is now only a fallback for custom profiles that omit it).
- **Removed `large` and `thorough`.** `optimize_parameters: "large"` / `"thorough"` now
  returns `400`. They made the set incoherent: `large` was *cheaper* than `auto` despite
  the heavier name, and `thorough`'s 10-candidate grid bought a measured **+0.0004**
  macro F1 — the C curve is flat, so paying 3.3× for grid width was never a real rung.
  Custom profiles in `config.yaml` remain freely definable for anyone who wants either.

  | Profile | Char n-grams | `C` grid | Evaluation | Head fits¹ | Deploys on |
  |---|:---:|---|---|---:|---|
  | `fast` | no | `[2, 32]` | holdout | 2.4 | 85 % of rows |
  | `auto` | (5,5) | `[2, 8, 32]` | 3-fold CV | 7 | **100 %** |
  | `best` | (5,5) | `[0.5, 2, 8, 32, 128]` | 5-fold CV | 21 | **100 %** |

  ¹ `(folds − 1) × |C_grid| + 1`, in units of the full dataset.
- **`auto` is now ~46 % cheaper than before** (7 head fits instead of 13) and still meets
  the goal that drove the redesign: the deployed model trains on 100 % of the rows and
  every row is scored out-of-fold. Fold count does not change *whether* the evaluation is
  honest, only how much data the evaluation models see (67 % at k=3 vs 80 % at k=5) —
  i.e. fewer folds bias the reported score slightly **pessimistic**.
- **The `C` grid is now part of the gradation too**, along the axis that turned out to
  matter. Two qualities of a grid are independent: *resolution* (candidate count inside
  the span) and *span* (how far out it reaches). Resolution is measured to be cheap —
  3 candidates at 4× steps select the same `C` at identical F1 as 8 candidates at 2×
  steps — so `best` spends its extra candidates on **span** instead: `[0.5, … , 128]`
  turns both known optima into *interior* points. That answers a question neither
  cheaper rung can: both targets currently pick a grid *endpoint*, which by this
  project's own rule means "the search ran out of candidates", not "it found an
  optimum". Pinned by `test_c_grids_step_up_along_the_ladder`.
- **Fixed a latent defect this surfaced:** `fast`'s grid was `[4, 16]`, which brackets
  *neither* known optimum (subject picks 32, educational level picked 2) — it would have
  shipped a model regularized at the wrong end while looking like a normal run. Every
  shipped grid now spans `2 … 32` at minimum, pinned by
  `test_every_profile_brackets_both_known_optima`.
- `GET /train/profiles` reports each profile's `cv_folds`; the UI's evaluation dropdown
  says "Profile default" and gained a 3-fold option.
- Docs corrected in the same pass: the README sizing section still described the removed
  `large` profile and rested on the superseded `(3,5)` character n-grams (725 nnz/doc);
  it now uses the `(5,5)` basis and labels derived figures as derived.

## prediction reliability, wider C search, field weights (2026-07-25)

Four changes from a review of where recognition quality is actually lost.
Test-first; 163 tests green, ruff/mypy clean.

### Decided — the head stays `LogisticRegression`
- A one-off benchmark compared `LinearSVC` (raw and calibrated) and a LogReg⊕SVM
  ensemble against the deployed head on `data_30k_ai.csv`. Quality was a tie within
  split noise, and once the probabilities the API is built on (`confidence`,
  `baseline_diff`, threshold grid, `/predict/multi`) are restored via
  `CalibratedClassifierCV`, LogReg wins on **speed, memory and probabilities** at once.
  The ensemble landed *between* its members because the two heads correlate at 0.9855.
  **Question closed; the benchmark code was removed rather than carried as dead
  exploration code.** The reasoning and all numbers survive in
  `docs/model-approach-comparison.md`.

### Added — `scripts/benchmark_label_scaling.py`
- Answers what changes for a wide vocab (`ccm:curriculum`, 478 labels available):
  48 vs 300 labels on identical texts/features/split. Everything is **linear** in the
  label count (fit ×5.79, coefficients ×6.25, threshold tuning ×5.65 for 6.25× labels),
  so a wide vocab is a budgeting question, not an architectural one. A 300-label head
  is 229 MB of float32 coefficients plus ~148 MB of vectorizer per model — worth
  planning the container limit around.
- 🟢 **The finding that matters:** at 300 labels micro F1 halves (0.6254 → 0.3191) and
  drops below macro. Recall barely moves (0.617 → 0.581) while **precision collapses**
  (0.634 → 0.220) because the model asserts **4.22 labels per row against a true 1.60**.
  Per-label F1-optimal thresholds buy recall with false positives on rare labels; macro
  dilutes that 1/300, micro pools it. This is the thresholding regime, not the
  classifier. Mitigations use what the API already has — `top_k` ranking mode,
  `include_label_f1`, and a higher `min_samples_per_label` for wide vocabs.

### Added — `label_f1` on predictions
- All predict endpoints can attach `label_f1` per prediction
  (`include_label_f1=true`; always on in `/predict/explain`): the label's F1 from
  the training evaluation. Confidence says how sure the model is *here*,
  `label_f1` how much that is worth — `0.95` on a label that only scores `0.68`
  overall deserves a human look. The value was already parsed on every model load
  and thrown away (`registry.get`); it is now carried on `ClassifierModel`.
  **Works with existing bundles — no retraining needed.** `null` for labels a
  bundle has no score for. The admin UI shows it next to the baseline diff.
- Malformed `per_label_f1` in an imported bundle's `metrics.json` (non-mapping,
  non-numeric, NaN) is dropped instead of reaching the model — it is reporting
  data arriving over a trust boundary.

### Added — `large` profile + a measured sizing envelope (up to 600 k rows)
- New `large` profile: word n-grams with a **full** 200 k vocabulary, 2 `C` candidates.
  🟢 Measured (`scripts/benchmark_row_scaling.py`, `benchmark_feature_caps.py`):
  memory and vectorization scale with **exponent 1.00** in the row count, so at 600 k
  rows `auto` needs a **3.2 GB** feature matrix and **24 min per head fit**, while
  word-only stays at ~340 MB and under a minute — for **−0.0072 macro F1**.
  The driver is non-zeros per document: **725 with character n-grams, 70 without**.
- `fast` is deliberately NOT the answer for big data: it is word-only *and* caps the
  vocabulary at 50 k *and* drops per-label thresholds, and the cap alone triples the
  quality loss (−0.0195 vs −0.0072). README gained a sizing table.
- **Fixed a scaling defect:** `prepare_targets` returned the dense 0/1 target matrix as
  `int64` — 8 bytes per bit. Now `int8`: 1.34 GB → 168 MB at 600 k rows × 300 labels.
- Known limit, documented rather than redesigned: k-fold CV holds one
  `rows × labels` float32 buffer **per C candidate** (0.9 GB at 600 k × 48 × 8), so a
  holdout split plus the 2-candidate `large` grid is the configuration for that size.

### Added — per-target over-assertion in the metrics
- Every bundle's `metrics` now carries `predicted_labels_per_row` next to
  `true_labels_per_row`. F1 hides over-assertion, and how badly a model over-asserts
  depends on the **training target**, not the dataset: the same rows yield ~1.1 asserted
  labels for a 48-label subject vocab and ~4.2 for a 300-label curriculum vocab. Reading
  one target's number off another's was exactly the mistake this makes impossible.

### Changed — `text_column_weights` is now a config default (title + keywords 2×)
- `config.yaml` gains `preprocessing.text_column_weights`, shipped as title +
  keywords at 2× — the measured optimum. It applies when a `/train` request omits the
  field and is **narrowed to the columns that request trains on**, so a global default
  cannot break a CSV with different column names. A request mapping overrides it;
  `{}` trains unweighted. `GET /train/profiles` now reports
  `default_text_column_weights` and `default_min_samples_per_label`, and the admin UI
  pre-fills its per-column multiplier inputs from them.
- The bundle records the **effective** weights (what was applied), not what the request
  contained — a request that inherited the default is still self-describing.

### Added — `max_word_features` / `max_char_features` per training request
- The vocabulary caps were reachable only via env vars or a profile, i.e. not per run.
  They are now request fields (most specific wins: request → profile → settings),
  bounded at 2 000 000 because they are the main RAM lever. Both effective values are
  recorded in the bundle metadata, which is what makes `tfidf.n_features` interpretable:
  if it equals their sum, the vocabulary was **truncated**. On `data_30k_ai.csv` it does
  (80 000 word + 120 000 char, both saturated). Exposed in the admin UI behind an
  "advanced" disclosure.
- 🟢 **Measured** (`scripts/benchmark_feature_caps.py`): the caps really are binding —
  the natural vocabulary is 134 835 word + 255 268 char — but the discarded tail carries
  **nothing**. Doubling the caps exhausts the word vocabulary entirely and still lands at
  **−0.0005 macro F1**, while the head grows 36.6 → 68.6 MB and fit time nearly doubles.
  `max_features` keeps the most frequent terms and `min_df=2` already drops hapaxes, so
  what is cut is noise. The shipped 80 k / 120 k stays. The useful direction is *down*:
  40 k / 60 k costs only −0.0052 macro and **halves the head**, which matters for a wide
  vocab on a small host (a 300-label head is 229 MB at 200 k features, ~115 MB at 100 k).

### Added — `text_column_weights` in the train request
- Repeats a text column when the training text is assembled
  (`{"properties.cclom:title": 2}`), so short dense fields are not drowned out by
  a long description. Values `1…10`; keys must be among `text_columns` (a typo is
  a `422`, not a silent no-op). Recorded in the bundle metadata. The admin UI
  generates one multiplier input per selected text column.
- **Training-time only.** `/predict` takes one opaque string and cannot re-apply
  the weights, so a model trained with weights expects input assembled the same
  way; otherwise its tuned thresholds sit on a slightly different distribution.
- 🟢 **Measured** (`scripts/benchmark_field_weights.py`, identical rows/split/C grid
  per variant): `{title: 2, keyword: 2}` gives **+0.0145 macro F1 / +0.0052 micro**
  over unweighted — the largest single quality gain measured on this data, at
  **no cost** in matrix size, RAM or fit time (a repeat raises term counts, it does
  not add new terms). Two non-obvious results: boosting **keywords alone** already
  yields the whole macro gain, while boosting the **title alone lands 0.0049 BELOW
  baseline**; and 3× is worse than 2×, so the multiplier should not be pushed up.

### Changed — wider C grid, and the grid is now recorded
- `auto` searches `[0.25 … 32]` (was `[0.5 … 16]`), `thorough` `[0.1 … 64]`.
  Motivation: `faecher_ai_cv5` selected `16.0`, the old grid's **maximum** — a
  pick at the edge means the search ran out of candidates. Costs ~33 % more fits
  in `auto`; the selection only keeps an extreme `C` when it scores better.
- 🟢 **Measured outcome, so nobody assumes this bought quality:** on
  `data_30k_ai.csv` the validation curve is *flat* above the old boundary —
  `C=16` scored 0.7271 macro F1, `C=32` scored 0.7275 (**+0.0004**). The widening
  buys certainty that the optimum is inside the grid, not accuracy. Widening
  further is not worth the fits.
- Bundle metadata gains `c_grid`, so `best_C` stays interpretable after the fact.

### Changed — `min_samples_per_label` defaults to 20 (was auto-scaled)
- The request field now declares `20` instead of silently scaling to dataset size,
  because dropping labels is a decision worth seeing. `null` still asks for the
  heuristic (2 / 5 / 20 / 35). The admin UI exposes the field.
- **Behaviour change for small datasets:** a dataset whose labels have fewer than
  20 rows now aborts unless the caller lowers the value. The abort message names
  the gap ("of 3 labels the most frequent one has only 12 tagged rows, below
  min_samples_per_label=20") instead of just restating the rule, and surfaces as
  `status=error` on `/train/status`.

## deferred re-audit items resolved (2026-07-16, late evening)

Closes the four items the same-day re-audit deliberately deferred
(B9/B10/C3/D10 in `docs/audits/2026-07-16-reaudit.md`). All test-first;
152 tests green (96% line coverage), ruff/mypy clean.

### Changed — metrics measure the serving decision rule (B9)
- For **multiclass/binary** models, C-selection and all reported metrics now
  measure the **argmax** rule serving actually applies (`ClassifierModel.predict`
  returns the single best label and ignores thresholds for single-label tasks).
  Previously they measured a thresholded rule serving never used, so reported
  F1/precision/recall could diverge from live behaviour in both directions.
  Threshold tuning is skipped for these models (neutral `0.5`/`{}` in the
  bundle instead of tuned dead values). Multilabel models are unchanged.
- **Reported numbers shift for single-label models**: retraining the same data
  can now yield different (honest) metrics. Bundles are self-describing via a
  new `decision_rule` key in `metrics` (`"argmax"` or `"thresholds"`); bundles
  trained before this change lack the key (= old thresholded numbers).

### Fixed — rows orphaned by the unlearnable-label drop (B10)
- The classic-split path drops label columns without train positives; rows
  whose ONLY labels were dropped stayed as all-zero targets — guaranteed misses
  in the metrics (fatal under argmax) that also diluted `avg_labels`. Such rows
  are now removed (split indices remapped) and `avg_labels` is computed AFTER
  the drop, so it describes the label space actually trained.

### Changed — **breaking**: `POST /datasets/{name}/validate` contract (C3)
- The endpoint now takes ONE JSON object like `/datasets/analyze`
  (`{"text_columns": [...], "label_column": "...", "csv_separator": ";",
  "label_separator": ","}`). The former mixed contract (raw JSON array body +
  query parameters) is gone; the separator's single-char guard moved into the
  schema. Update callers accordingly (the bundled admin UI never used this
  endpoint).

### Fixed — container-aware CPU budget (D10 code half)
- `effective_n_jobs()` now derives "all cores" from what the process may
  actually use: `os.cpu_count()` bounded by the Linux scheduler affinity mask
  and the cgroup CPU quota (v2 `cpu.max`, v1 `cfs_quota_us`; fractional quotas
  floor, never below 1). In a 4-CPU-limited pod on a 64-core node, `-1` now
  means 4 instead of ~38 throttled threads. The Helm default returns to
  `nJobs: -1` (it follows a resized `resources.limits.cpu` automatically), and
  `APIV3_CPU_MAX_PERCENT` finally means what it documents inside containers:
  training keeps ~40% of the POD's quota free for serving.

## re-audit fixes (2026-07-16, evening)

Acting on the same-day re-audit (`docs/audits/2026-07-16-reaudit.md`; four
fresh-eyes readers + scanners). 134 tests green (95% line coverage), ruff/mypy
clean; all fixes test-first where logic.

### Fixed (correctness)
- **Per-label threshold degeneracy**: a label with zero positives in the val
  split got the grid MINIMUM (0.05) instead of the global-threshold fallback —
  rare labels then fired on much of the traffic. Zero-positive labels now keep
  the global threshold.
- **Registry cache/delete race**: a cold `get()` could re-insert a concurrently
  deleted model into the LRU cache (a same-name re-import then served the OLD
  weights). The miss path AND `delete()` now change disk + cache atomically
  under the disk lock.
- **Zombie-thread state pollution**: after `POST /train/stop?hard=true`, the
  abandoned thread's progress updates mutated the reset state (`/metrics` showed
  `training_running 0` with progress creeping). Progress updates now carry the
  same generation guard as completion.
- **Two-trainings window closed**: the runner thread is registered in the same
  lock block as the state transition (a hard-stop + start in the microsecond gap
  could previously pass the overlap guard).
- **Import-vs-zombie-save guard**: model import refuses a name that a still-live
  training thread is writing — also after a hard stop reset the status to idle.
- `/train/status` now populates the documented `error` field on failures.
- `validate` counts "rows without labels" from the raw label column (the loader
  drops such rows, so the documented warning could never fire before).
- Corrupt `config.json` / wrong-shaped bundle content → 422/400 instead of 500.

### Security / API
- **Single-char separator guard on ALL inputs** (`/train`, `/datasets/analyze`,
  `validate` — previously only `GET /datasets/{name}`): pandas parses a
  multi-char separator as a regex (ReDoS; in `/train` it could hang the training
  thread outside any stop checkpoint).
- `validate` maps malformed-CSV errors to a crafted 400 like its siblings
  (was a sanitized 500) and documents its array-body + query-param contract.
- Swagger `/docs` no longer persists the API key in localStorage
  (`persistAuthorization` removed — the admin UI's sessionStorage-only posture
  now holds for both entry points).
- Unknown-profile 400 detail no longer arrives wrapped in stray quotes.

### Frontend
- Share box: inline `onfocus` handler (blocked by the UI's own CSP) replaced
  with an addEventListener — plus a test pinning "no inline handlers" at source
  level. Bundle-controlled `metrics.json` values are now escaped and coerced
  before rendering (stored-HTML-injection / table-crash path via model import).
- Dark mode: destructive buttons use a `--danger-text` token (was white on
  light salmon, ≈2.4:1). Pill inputs got accessible names; the training status
  card is no longer a live region (a dedicated SR-only element announces phase
  TRANSITIONS instead of re-reading the card every 2.5s). Label-field pills
  join the pillbox flex layout; dead `pattern` attribute removed; the status
  poll is re-entrancy-guarded and the client queue also advances past an
  externally caused `idle`.

### Performance
- Admin model routes (`delete`, `export`, `import`, share download) and the
  dataset-import write run via `asyncio.to_thread` — a multi-second zip/skops/
  rmtree no longer freezes `/health` and predicts on the single worker.

### Tests / Deps / CI
- Suite is hermetic against ambient `APIV3_*` env vars and a local `.env`;
  fresh-app fixtures also reset the registry singleton; module temp dir is
  reclaimed at exit; warmup corrupt-bundle branch, `/train`+`/train/stop`
  response key sets and all `/docs` runtime dependencies are now pinned by tests.
- **`joblib` declared as a direct dependency** (13 pins; it was imported
  directly but rode along transitively — invisible to the lock-parity gate).
  `types-PyYAML` added to the pyproject dev extra; ruff/mypy targets bumped to
  py311 (matching the floor); `scripts/` joined the lint gate.
- `.dockerignore` cache patterns made recursive (`**/__pycache__`); GitLab CI:
  helm image pinned (was `latest`), `docker login` via `--password-stdin`,
  branch chart pushes restricted to main/develop like the image jobs.
- Helm `values.yaml`: `nJobs` default now matches `resources.limits.cpu` (the
  app derives "-1 = all cores" from the NODE, not the CFS quota — documented).

## audit remediation (2026-07-16)

Acting on the whole-codebase audit (`docs/audits/2026-07-16-audit.md`). All
findings addressed except three deliberately deferred (see that report's status
banner). 118 tests green, ruff/mypy clean.

### Security
- **Strict Content-Security-Policy** on every response (`default-src 'self';
  base-uri 'self'; form-action 'self'; frame-ancestors 'none'; object-src 'none'`)
  — a second XSS layer for the same-origin admin UI, mirroring the data-prep
  sibling (audit T8). Verified live (all UI assets + JS executed, zero violations).
- **Self-hosted API docs** — `/docs` now serves vendored Swagger UI assets
  (`swagger-ui-dist@5.17.14`) with an external init script instead of the CDN +
  inline-init default, so it is same-origin and runs **under CSP** (a `/docs`-only
  relaxation allows `style-src 'unsafe-inline'` + `img-src data:` that Swagger
  needs). The CDN-backed ReDoc (`/redoc`), redundant with Swagger, is disabled.
  Verified live: 24 operations render, zero CSP violations, no external requests.
- **`safe_name` now rejects `:`** — Windows drive-relative (`D:x`) and NTFS
  alternate-data-stream (`x:stream`) names escaped the storage dir (audit T4).
- **Non-ASCII `X-API-Key` → 401**, not a `TypeError`-driven 500 (`secrets.compare_digest` raises on non-ASCII).
- **CORS**: a wildcard origin no longer combines with credentials (guarded).
- **Rate limiting**: `--proxy-headers` in the image so limits key on the real
  client IP behind a proxy (audit T3); `DELETE` model/dataset are now throttled
  and the previously-dead `rate_limit_default` is wired; `separator` was capped to
  one character on `GET /datasets/{name}` (ReDoS — the evening re-audit extended
  this guard to every separator input); the 429 body now uses the shared
  `{"detail": ...}` envelope.

### Fixed
- **Hard-stop no longer allows two concurrent trainings** (audit T2): a new
  `/train` is refused while the hard-stopped thread is still finishing.
- **Non-UTF-8 CSVs load** via a cp1252 fallback, and empty/malformed CSVs return
  400 instead of 500 (audit T7); `dataset_info` now delegates to `data.sample_rows`.
- **Too-few-rows-after-dropping-rare-labels** is rejected instead of "succeeding"
  on a near-empty split (audit T10).
- **ShareStore** writes atomically (tmp + rename) and skips a single malformed
  entry instead of resetting/crashing the whole store.
- **Registry cache/disk race** on save closed (cache insert under the disk lock).
- **`sweep_stale_tmp` no longer over-counts**: with `rmtree(ignore_errors=True)` a
  removal can silently fail (locked file); the returned/logged count now reflects
  dirs actually gone, not attempts.
- Removed the dead, no-effect `use_auto_settings` predict field (extra fields are
  ignored, so clients still sending it are unaffected) and corrected the stale
  `/predict` docstring that still described the removed auto-`top_k` behavior.

### Performance
- **Startup warmup** (`APIV3_WARMUP_MODELS`): named models are preloaded into the
  LRU cache and run one empty-text prediction on startup, so their first real
  `/predict` pays no cold skops-load. Best-effort — a missing/unreadable name is
  logged and skipped, never fatal to startup; the resolved list shows in `/config`.
- **Model cold-loads run off the event loop** (audit T5): `_load` moved inside
  `asyncio.to_thread` for predict/multi/explain.
- **Long skops save no longer holds the disk lock across the dump** (audit T6):
  a `/predict` of another model is not blocked for the whole save.

### Deployment & Ops
- **Helm: writable `/tmp` emptyDir** so multipart uploads > 1 MB no longer fail
  under `readOnlyRootFilesystem` (audit T1); README claim corrected.
- **Helm hardening**: `automountServiceAccountToken: false`, `seccompProfile:
  RuntimeDefault`, PDB `maxUnavailable: 1` (single-replica node drains).
- **CI**: top-level `permissions: contents: read` on both workflows; all GitHub
  Actions pinned to immutable commit SHAs (with `# vN` comments), plus a
  `dependabot.yml` (github-actions ecosystem) to keep those pins current. Pinning
  is behavior-identical to the prior tags (same commit) but immune to tag-moving
  supply-chain attacks. *Not runnable here — confirm on the first CI run.*

### Changed
- **Renamed to MetaClassify** in all user-facing text (OpenAPI title/description,
  self-hosted `/docs`, admin UI, README, docs, Helm/Chart/image descriptions,
  LICENSE). Internal identifiers are intentionally unchanged — the `api_v3`
  directory, `app` package, `APIV3_` env prefix, Helm chart name and container
  image path keep their names so existing deployments and config keep working.

### API
- **Response models** on the fixed-shape endpoints (`/health`, `/config`,
  `POST /train`, `/train/stop`) document the contract in OpenAPI (new `responses.py`).
  Endpoints with conditional/dynamic keys (predict results, `/train/status`,
  dataset analyze/validate) deliberately keep `dict` returns — a `response_model`
  filters output to its fields and would silently drop keys those responses omit
  or add situationally (e.g. predict omits `baseline_diff` when not requested but
  intentionally keeps a null `top_k`, which no single `exclude_none` policy fits).

### Refactor (behavior-preserving)
- Split `data.py` (was 338 lines): read-only dataset inspection/statistics
  (`sample_rows`, `analyze_dataset`, `validate_dataset`) moved to `dataset_stats.py`,
  which only consumes the load/clean/target-prep core. The shared CSV reader
  `_read_csv` is now public `read_csv`.
- Split `registry.py` (was 352 lines): the security-critical pickle-free bundle
  (de)serialization + skops load-safety moved to `model_io.py`; `registry.py`
  keeps the LRU cache, two-lock disk orchestration, atomic publish and zip
  import/export. Both files are now under the 300-line guideline.

### Testing / Deps / Docs / Frontend
- Autouse fixture resets the process-global rate limiter between tests (removes
  latent 429 flakiness); added `/config`, CORS, separator, 429-envelope, and
  hashes-present tests.
- `requires-python` raised to `>=3.11` (matches the Docker/hashes target);
  `types-PyYAML` pinned in `requirements-dev.txt`.
- UI: the `visibilitychange` handler is registered once (was leaking a listener
  per logout→login cycle).
- UI: the section tabs now follow the WAI-ARIA tabs pattern — `role="tablist"`/
  `tab`/`tabpanel`, `aria-selected`, `aria-controls`/`aria-labelledby`, roving
  `tabindex`, and Left/Right/Home/End keyboard navigation (was `aria-current`,
  the nav-link idiom, with no arrow-key model).

## hardening & evaluation wave (2026-07-07 … 2026-07-08)

### Added
- **UI feature completion**: share links for models AND datasets (with copy
  button; served key-less via `GET /share/{id}`), model import (ZIP), dataset
  download; multi-field training — pick several label fields, the UI queues
  one `POST /train` per field sequentially (single-worker server) and derives
  names as `base_field` with a live preview. `/ui` assets are served with
  `Cache-Control: no-cache` so browsers never run a stale UI after updates.
- **Plain-language help**: every UI tab has a collapsible jargon-free help
  section (confidence, top-k modes, baseline diff, F1 micro/macro, CSV format,
  share links); plus `docs/ui-guide.md` — a non-technical walkthrough for
  editorial users (German, matching its audience) incl. FAQ.
- **Keyless UI mode**: the admin UI mirrors the server's auth setting like
  `/docs` — with `APIV3_AUTH_ENABLED=false` it skips the sign-in ("auth
  disabled" badge) instead of demanding a key the server does not require.
- **Configuration reference** at `docs/configuration.md`: every `APIV3_*`
  variable with defaults, `config.yaml` options, per-request overrides, and
  local/production quick setups; `.env.example` completed
  (`APIV3_CONFIG_FILE`, keyless note).
- **Granular training progress**: the C search (55→75 %) and cross-validation
  (45→90 %, one step per head fit with `Fold i/k: C=… (n/total fits)` detail)
  now advance continuously, making the ETA meaningful — previously a 30k CV
  run sat at a frozen 45 % for ~25 minutes while the ETA grew. The admin UI
  shows a training chip in the top bar, visible on every tab.
- **Admin UI at `/ui/`** — self-contained static page (vanilla JS, no build
  step, no CDN assets): slim API-key sign-in bar (sessionStorage, X-API-Key
  header), dataset upload/management, training with live progress + stop
  (text columns picked via a type-ahead pill selector backed by a native
  `<datalist>`), model list with metrics/export/delete, and test queries incl.
  multi-model + `baseline_diff`. Same-origin (no CORS), WCAG-AA color tokens
  with dark mode, keyboard-operable. Disable via `APIV3_UI_ENABLED=false`.
- **CPU budget for training** — `APIV3_CPU_MAX_PERCENT` (default **60**): the
  effective head-fit thread count is `min(n_jobs, cores × percent/100)`, so a
  training run leaves ~40 % headroom for the API (measured: 55 % peak on a
  16-core machine). `100` disables the cap; `GET /config` shows the resolved
  `effective_n_jobs`.
- **`baseline_diff` diagnostic** (idea adapted from `openeduhub/its-jointprobability`):
  confidence minus the model's empty-text prediction per label, separating the
  text's contribution from the label's base rate. Opt-in on all predict
  endpoints (`include_baseline_diff`), always on in `/predict/explain`.
- **`POST /predict/multi`** — classify each text with several models (= target
  fields) in one call. Orchestration only: each model applies its own tuned
  thresholds and keeps its own per-model evaluation metrics.
- **k-fold cross-validation** as an alternative to the train/val/test split
  (`split.cv_folds` in `config.yaml` or `cv_folds` per training request):
  out-of-fold metrics on every row, deployed model fit on 100 % of the data.
- **`GET /metrics`** — Prometheus text-format gauges (uptime, models on disk /
  in the LRU cache, training running + progress), public like `/health`; the
  Helm ServiceMonitor scrapes it.
- **`requirements-hashes.lock`** — full transitive dependency tree with sha256
  hashes (compiled for the image's Python 3.11); Docker and CI install with
  `--require-hashes`. Guarded by a lock-consistency test.
- **Baseline security headers** on every response (`nosniff`, `X-Frame-Options:
  DENY`, `Referrer-Policy: no-referrer`).
- `TrainingInputError`: crafted, safe messages (wrong column name, too few
  rows, bad `cv_folds`) surface on `/train/status` and as 400 on
  `/datasets/analyze`; all other failures stay sanitized with server-side logs.

### Fixed
- Text cleaning (`clean_text`) is applied at **prediction** time (all predict
  endpoints), matching training — no train/serve skew.
- CV mode no longer drops rare labels whose positives fall outside the (unused)
  train split.
- Registry disk I/O is serialized (`_disk_lock`); training injects the shared
  registry singleton; a model deleted mid-request returns 404 instead of 500.
- `/train/stop` honours its "between the C fits" promise in CV mode; a hard
  stop can no longer be overwritten by the finishing training thread.
- Rate limits cover every heavy or public route (CSV-reading dataset endpoints,
  key-less `GET /share/{id}`); CSV row counts are cached by (mtime, size).
- Orphaned model staging dirs (a hidden `.name.tmp` left when a save is
  crash/OOM-killed mid-write) are swept on startup (`Registry.sweep_stale_tmp`),
  so a killed training no longer leaks disk until the same name is retrained. The
  atomic-write guarantee already kept such partials out of the registry; this
  only reclaims the space.

### Changed
- **`top_k` semantics redesigned** (field looked "dead" in practice): without
  `top_k` the model **decides** — multilabel via its tuned per-label thresholds
  (the old `round(avg_labels)` auto-cap is gone; it silently swallowed
  legitimate second labels), multiclass via argmax. An explicit `top_k=N` is a
  **ranking**: exactly the N most probable labels regardless of thresholds
  (works for every task type, e.g. top-5 candidates on multiclass), each
  flagged `above_threshold`; the UI dims entries below their threshold.
  `use_auto_settings` is deprecated (no effect).
- Training pipeline split by stage: `prepare.py` → `deploy.py` → `training.py`;
  explainability extracted to `explain.py`; routes stay thin.
- Docker base image digest-pinned; Helm runs with
  `readOnlyRootFilesystem: true`; CI lints the Helm chart and installs the
  exact hashed dependency tree; dev tooling pinned (`requirements-dev.txt`).

## [3.0.0] — torch-free rewrite

Initial v3: FastAPI + TF-IDF (word+char, sparse float32) + OneVsRest
LogisticRegression, skops (pickle-free) persistence, two-role API-key auth,
file-upload-only imports (no SSRF), single-worker design with LRU model cache.
See `docs/audits/` for the audit trail.
