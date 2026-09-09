# Ergänzungsplan — MetaClassify (api_v3), 2026-09-08

**Status: DRAFT — awaiting owner approval.** Companion to
`docs/audits/2026-09-08-audit.md` (finding ids F1, O1, S1 … refer to it).

## Goal

Take api_v3 from "correct and hardened" to a tool the editorial team uses daily, along
the four owner goals — **fast on CPU, short training runs, shareable models, good
recognition rate** — plus a UI that exposes what the API already knows. Every
performance and quality item below is a *measured experiment first, product change
second*: nothing lands without a benchmark number, the way the July work was done.

## Global constraints (from CLAUDE.md, unchanged)

- English comments/docstrings/API texts; test-first for every fix; never weaken a test.
- No pickle (skops); import via upload only; `security.safe_name` for every user name;
  constant-time key compare.
- **13 runtime dependencies stay 13.** Every item below is stdlib + scikit-learn +
  existing deps. The i18n layer is a JSON string map in vanilla JS, not a library.
- Files ≤ ~300 lines; split by responsibility. Single-worker design stays.
- Bundle format stays **2** unless an item says otherwise (C1 adds a *member*, which is
  backward compatible: old readers ignore it, new readers verify when present).

## Scope

In: the items below. Out: multi-worker/queue systems, embeddings or transformer
backends (measured in July to lose to TF-IDF here), a JavaScript build toolchain, any
new runtime dependency, a database.

---

## Phase 0 — Stop the bleeding (≤ 1 day) — **DONE 2026-09-09**

> Published to `github.com/janschachtschabel/meta-classification` and released as
> **v3.1.0**. 0.1, 0.2 and 0.4 are complete; 0.3 is complete in code, and its two
> data items are settled below. 198 tests, ruff, mypy and pip-audit clean; CI and
> the Docker build green on GitHub Actions.
>
> **Model inventory checked (the open question in 0.3).** Of the four format-1
> bundles, exactly one still has no successor:
>
> | target | newest bundle | rows | format | verdict |
> |---|---|---:|:---:|---|
> | subjects (`taxonid`) | `faecher_300k_auto` (2026-07-26) | 156,373 | 2 | successor exists — no retrain needed |
> | educational level (`educationalcontext`) | `bildungsstufe_ai_cv5` (2026-07-19) | 25,763 | 1 | **no successor — retrain still open** |
>
> `faecher_cv5` and `faecher_synth_bal` are superseded experiments. Note the macro
> figures of the two subject models are *not* comparable (48 vs 59 labels, different
> data and evaluation), so "the new one is better" is not a claim this table makes.
> Training is deferred at the owner's request (machine load).
>
> **Added beyond the plan** (owner request, same session): every model listed in
> `APIV3_WARMUP_MODELS` now stays resident — the cache is sized to fit the list —
> and the setting is wired through `docker-compose.yml` and the Helm chart. It had
> existed only in `.env.example`, so the containerized deployment it is meant for
> could not configure it. 🟢 Measured with the cap deliberately at 2 and three real
> bundles warmed: 3 of 3 resident, every `/predict` ≈ 0.09 s.
>
> **Dependencies:** the grouped Dependabot PR (7 GitHub Actions, among them
> `actions/checkout` 4.3.1 → 7.0.1 and `setup-python` 5.6.0 → 7.0.0) was merged
> after its own checks passed; the stale `# v4` comment it left on the new SHA was
> corrected. Python runtime dependencies were **not** touched: `pip-audit` reports
> no CVEs, so there is no driver, and bumping them would mean recompiling the hash
> lock and re-validating the model pipeline for no gain.

| # | Item | Files | Acceptance |
|---|---|---|---|
| 0.1 | **Commit, push, first CI run.** Six logical commits (format-2 · gzip · label repair · container guard + profiles · docs · tests); add remote; tag `v3.1.0`. Owner decides `data/label_names.json` (recommend: commit as dated snapshot). | `.gitignore` (+`char_range_results.json`, `fertige-modelle/`, `.claude/`), `CLAUDE.md` (git note) | `git log` shows the slices; GitHub Actions green on push (ruff/mypy/pytest/OpenAPI smoke); image built by `docker.yml` |
| 0.2 | **F1 — CSP-safe bars.** Templates emit `data-width`; one helper applies `el.style.width` after each `innerHTML` render (three sites). | `app/static/ui/app.js` L149, L221–222, L251 (+ helper) | Live: `width:0%` bar measures 0 px; console has zero CSP errors on Query + Training tabs. Test: `test_ui_sources_contain_no_inline_style_attributes` (grep for `style="`/`` style=` `` in `app/static/ui/*.js`) — red before, green after |
| 0.3 | **O3 — unloadable bundles.** `registry.list()` skips `*.prebackup`; `info()`/list expose `format_version`; UI greys rows with `format_version < 2` and says "retrain". Retrain `faecher_ai_cv5` + `bildungsstufe_ai_cv5` with `best`; delete `faecher_cv5`, `faecher_synth_bal`, the three `probe_*` and both `.prebackup` dirs after the owner confirms. | `app/registry.py`, `app/routes/models.py`, `app/static/ui/manage.js`; `scripts/prune_bundle_labels.py` writes backups as `.<name>.prebackup` (hidden) | Test: a dir named `x.prebackup` is absent from `GET /models`; a format-1 fixture bundle reports `format_version: 1` and renders greyed |
| 0.4 | **Quick wins.** S1 `_dataset_path()` (suffix-checked, shared by info/export/share/delete) · S3 name length ≤ 100 · S4 startup refuses auth-on-without-admin-key, warns on identical keys · S5 `.env` anchored to `_BASE` · C1 idle hard-stop is a no-op · A3 `POST /train` → 202 · A5 `GET /` → 307 `/ui/`, favicon (inline SVG data route) · D2 `training_job.stop()` at shutdown + Helm comment · X1 weekly `pip-audit` job · D1 three doc fixes | `app/routes/datasets.py`, `app/security.py`, `app/settings.py`, `app/main.py`, `app/jobs.py`, `.github/workflows/ci.yml`, `README.md`, `docs/configuration.md` | One test per behaviour change (S1: delete/export/share of `label_names.json` → 404; S4: `Settings(auth_enabled=True)` without keys raises; A3: 202) |

Effort: 1 person-day. Risk: none of these changes model behaviour.

---

## Phase 1 — Shareable models + honest model view (1–2 weeks)

### C1 · Bundle manifest + model card (S–M, 1–2 days) — **done 2026-09-09**
- `manifest.json` member: `sha256` per file, `app_version`, `format_version`,
  `created_at`, `n_labels`, `f1_macro`, `dataset`. Written by `_write_bundle`; verified
  by `_read_bundle` when present (mismatch → `UnsafeModelError`, i.e. 400 on import,
  422 on load). `README.md` member: a generated model card (dataset, columns, weights,
  profile, C/grid, folds, metrics, the 10 weakest labels, a curl example).
- **Why:** a share link moves a 50–180 MB ZIP between servers; today a truncated or
  tampered member is only caught if skops chokes. A card makes the ZIP self-explaining
  for the recipient.
- Files: `app/model_io.py` (+ `app/model_card.py`, new, < 120 lines), `app/registry.py`
  (`_ALLOWED_MEMBERS` + 2). Tests: round-trip keeps hashes; a flipped byte in
  `head.skops` is rejected on import; an old bundle without manifest still loads.

### C2 · Share-link management (S, ½ day) — **done 2026-09-09**
- `GET /share` (admin: id, kind, name, expires_at, downloads) and `DELETE /share/{id}`
  (admin); optional `max_downloads` on create. UI: "Active share links" box on the
  Models and Datasets tabs with Revoke.
- Files: `app/sharing.py` (+`list`, `revoke`, `count_download`), `app/routes/models.py`,
  `app/static/ui/manage.js`. Tests: create → list → revoke → `GET /share/{id}` 404.

### C3 · Streamed transfers (S, ½ day) — **done 2026-09-09**
- `export_zip` writes to a `tempfile` and returns a `FileResponse`; dataset uploads
  spool to `<name>.part` and `os.replace`. Tests: a crash between write and replace
  leaves no listed dataset.
- 🟢 Measured on the real 51 MB `faecher_300k_auto`: export peak heap **142 MB → 2.2 MB**
  (2.78× the bundle → 0.04×). The old path held every member, the zip built beside them,
  and the copy `getvalue()` makes. Cost: compression now runs under the disk lock, 3.1 s
  for that bundle — the members are read one at a time from files that must still exist,
  and holding open handles outside the lock would make `delete` fail on Windows.
- Staging goes next to the bundles, not into the system temp: on a container `/tmp` is
  often tmpfs, i.e. RAM, which would give back exactly what this removes. It carries the
  hidden `.*.tmp` name the startup sweep already cleans (extended from dirs to files).
- **Model import still buffers** (measured 172 MB peak, 3.37×): `unpack` validates the
  archive as bytes before anything reaches the filesystem, and that order is the security
  property. Streaming it means extracting to staging first — a separate change.

### B8 · Per-label diagnostics endpoint (S, ½ day) — **done 2026-09-09**
- `GET /models/{name}/labels` → `[{uri, label, f1, threshold, support}]` sorted by F1.
  `support` (positives in training) is added to `metrics.json` at train time
  (`compute_metrics` already has `y_true`). Feeds D1 and the model card.
- Files: `app/tuning.py` (support), `app/routes/models.py`. Test: sorted, complete,
  `support` equals the column sums of the fixture.

### D1 · Model detail view in the UI (M, 2 days) — **done 2026-09-09**
- Click a model row → panel: metadata (dataset, text columns + weights, profile, C +
  grid with an "edge pick" hint, folds, rows, duration, created), the label table from
  B8 (sortable, weakest first, threshold shown), actions (download, share, delete,
  copy-curl). Format-1 rows show the retrain hint (0.3).
- Files: `app/static/ui/manage.js` → split into `share.js` (75), `models.js` (145),
  `datasets.js` (51) and `model-detail.js` (185), `index.html`, `style.css`. Four
  files, not the two planned: share links are used by BOTH tabs, and the detail view
  is a second view, not more of the list.
- Deviation: the trigger is the model NAME, not the row. A `<tr>` is not focusable
  and would fight the action buttons it contains; a button in the name cell is
  reachable by keyboard and does not.
- Verified in the running UI: focus trap proven (focusing an element outside the
  modal is refused), tab order Close → 4 sort headers → 5 actions, sorting keeps
  unscored labels last in both directions, 360 px without horizontal scrolling,
  every target ≥ 24×24, dark mode via the tokens, no console/CSP errors.

### P1 · Shared-vectorizer cross-validation (S–M, 1 day incl. benchmark) — **measured 2026-09-09; default unchanged**
- `cross_val_evaluate` accepts a prefitted matrix: `fit_transform` once over all rows,
  slice per fold, reuse for the deploy fit. Profile flag `refit_vectorizer_per_fold`
  (default decided by the benchmark). `scripts/benchmark_shared_vectorizer.py` reports
  macro/micro F1 and wall-clock for both modes on `data_30k_ai.csv` and the
  university target.
- **Unblocked 2026-09-09 (short runs):** the owner allows benchmark runs on
  `data_30k_ai.csv` (minutes per pass). The large 156 k / 426 k runs stay off the table,
  which is fine — P1's gate measures on 25–30 k rows by design. Nothing else depends on
  it: the flag only changes a default, and today's default is the measured-safe one
  (refit per fold).
- 🟢 **Measured on `data_30k_ai.csv`** (26 450 rows × 48 labels, `auto` shape): shared
  macro **0.7337** vs refit **0.7320** (+0.00171), **6.1 %** of the CV phase saved. The
  delta is inside the gate but uses 86 % of its budget, in the direction leakage predicts.
- ⛔ **The second target could not be measured and the default therefore stands.**
  `data_30k*.csv` carries ONE row with a `hochschulfaechersystematik` label, and the
  300 k export the university models were trained from is not on this machine. The gate
  says *both* targets; half the evidence is not the gate.
- Delivered: the mechanism (`cross_val_evaluate(matrix=...)`), the profile flag
  `refit_vectorizer_per_fold` (True everywhere), the benchmark script. To adopt: obtain
  the university export, rerun the script, flip the flag in `config.yaml`.
- Note: the measured 6.1 % covers the CV phase only. Reusing the shared matrix for the
  deploy fit as well — the rest of the plan's ~17 % estimate — is a further change with
  its own measurement to make.
- **Gate:** adopt as default only if |Δ macro F1| < 0.002 on both; expected saving
  ≈ 17 % (`auto`) / 20 % (`best`) at 156 k rows (README: 2.3 min per pass).
- Files: `app/tuning.py`, `app/deploy.py`, `app/profiles.py`, `config.yaml`, README
  (measured table). Tests: both modes produce identical OOF shapes; the flag is
  recorded in `metrics.json`.

### A1 · Split `data.py` (S, 1 h) — **done 2026-09-09**
- Move `_rejoin_split_names`, `_pair_names` and `label_vocabulary` to
  `app/label_names.py`; behaviour-preserving, tests move with them.
- `is_container_label` went with them (not in the original list): it reasons about the
  same "/" structure as `label_vocabulary`, and `model_io` + the repair script were
  already importing it *through* `data`, pulling pandas and sklearn behind it.
  `_pair_names` became public `pair_names` — it is now `data`'s dependency, and
  importing an underscored name across modules is not an interface.
- data.py: 395 → 309. Still a hair over the ~300 guideline and deliberately left there:
  what remains (load, clean, targets, split) is one responsibility, and cutting it
  again to satisfy a number is what the guideline warns against. 222 tests unchanged.
- ~~`registry.py` crossed the ~300-line guideline~~ **done 2026-09-09**: the archive
  half moved to `app/model_archive.py` (pure byte functions) when the card and manifest
  pushed the file to 343 lines. Registry was back to 279 and keeps only what needs the
  disk and the locks; the same 207 tests passed unchanged before and after.
  The reporting half followed on the same day: `info` and `label_diagnostics` grew back
  to 328 lines as the label report and the metadata guards landed, so they moved to
  `app/model_report.py` (pure functions over the two documents, the seam `data` /
  `dataset_stats` already uses). Registry: 293. That split also removed the second,
  unlocked parse of `config.json` the label report was doing.

Effort: 7–8 person-days. Risk: P1 is the only item that can change numbers, and it is
gated.

---

## Phase 2 — The UI catches up with the API (2–4 weeks) — **DONE 2026-09-09**

### D2 · Batch classification (M, 2–3 days) — **done 2026-09-09**
- Query tab: "Many texts" mode — paste one text per line *or* upload a CSV, choose
  the text column(s) and the weights the model expects (read from the bundle), run
  in chunks of 1000 through `/predict/batch`, show a table, download the result as
  CSV (`uri`, `label`, `confidence`, `above_threshold` per row). Server: a
  `POST /predict/csv` endpoint that takes an uploaded CSV and streams a CSV back
  (bounded by `max_upload_mb`; chunked, no full-file materialisation).
- **Why:** the daily editorial job is "classify these 500 new items", not one text.
- Files: `app/routes/predict_bulk.py` (new — the CSV route put a second responsibility
  into `predict.py` and pushed it past 300 lines), `app/predict_csv.py` (new),
  `app/static/ui/query.js` (split from `app.js`). Tests: 3-row CSV round-trip; the
  assembled text compared against what `load_dataset` builds for the same file; a row
  with no label still emitted; a label containing a comma; cp1252; oversize → 413.
- Deviation: the UI's pasted-text mode goes through `/predict` in batches of **500**,
  not 1000 — half the endpoint's cap, so progress is visible on a long paste. The CSV
  mode does not chunk client-side at all; the server streams it.
- 🟢 Measured: 50 000 rows against the 59-label `faecher_300k_auto` → 4.1 MB of CSV in
  54.6 s at a **2.2 MB peak heap** on a 6.5 MB input; `transfer-encoding: chunked`.

### D3 · Dataset inspector + training pre-flight (M, 2 days) — **done 2026-09-09**
- Datasets tab: click a row → columns, first rows (`GET /datasets/{name}`), and an
  "Analyze" button (`/datasets/analyze`) showing label counts, the `labels_with_N+`
  table and warnings. Training tab: after picking dataset + label field, show the
  same numbers inline with a recommended `min_samples_per_label` and an estimated
  duration from rows × profile (README cost model, labelled "estimate").
- Files: `app/dataset_stats.py` (+ the recommendation and the cost estimate),
  `app/profiles.py` (the cost model), `app/static/ui/dataset-detail.js` (new),
  `datasets.js`, and `app.js` split into shell / `training.js` / `train-status.js`.
  Tests: the cost model reproduces all six published README figures; analyze carries the
  fields the UI reads.
- Deviation: `support` per label was NOT added to `dataset_stats` — `analyze` already
  returns `top_20_labels` and `rare_labels_under_10`, and the threshold table answers the
  question the UI actually asks ("how many labels survive N"). A per-label support map
  over a 300 k dataset would be a large payload nothing reads.
- Deviation: both views analyse **on demand**, not on open / on change. Analysis parses
  every row; opening a panel or changing a select should not start a 195 MB job.

### D4 · Explain view (S–M, 1 day) — **done 2026-09-09**
- "Why?" on a prediction → `/predict/explain`; the top words per label rendered as
  chips with their impact; `all_scores` collapsible.
- Deviation: the button sits on the model's ANSWER, not on a single prediction line —
  one call explains the top predicted labels of that model at once, which is what the
  endpoint does. In multi-model mode each card gets its own.
- 🟢 Measured, and it changed the design: impacts are tiny on a confident answer
  (0.0021 against a 0.9997 confidence) and large on a contested one (0.5544). The bars
  are therefore scaled **per label**; a shared scale would render the confident case as
  an empty row. Negative impacts are common on weaker labels and are marked as arguing
  *against* the label.
- Files: `app/static/ui/explain.js` (new, 98), `query.js`, `index.html`, `style.css`.

### A5 · Server-side training queue + D5 persisted history (M, 3 days) — **done 2026-09-09**
- `POST /train` while busy → 202 with a queue position (bounded queue, e.g. 10);
  jobs persist to `models_dir/.jobs.jsonl` (request, status, metrics, duration);
  `GET /train/history`; the UI's tab-bound queue goes away (the browser can close).
  `TrainingJob` becomes `JobRunner` with `kind` (training | evaluation) so B1 reuses it.
- Deviation: the rename to `JobRunner` with `kind` is **deferred to B1**. Generalising a
  concurrency-critical class before its second use case exists is speculative, and B1
  will show what the second kind actually needs.
- Deviation: the queue does not persist across a restart. What persists is the HISTORY
  of finished runs; a waiting run has produced nothing yet, and silently resuming work
  a restart interrupted is a surprise, not a feature.
- **Why:** "train five label fields overnight" currently needs an open tab; history
  is the only way to compare runs without opening bundles.
- Files: `app/jobs.py` (split: `jobs.py` runner + `job_history.py`), routes, UI.
  Tests: queue order, persistence across a fresh `JobRunner`, hard stop clears the
  queue.

### B1 · Evaluate an existing model on a dataset (M, 2 days) — **done 2026-09-09**
- `POST /models/{name}/evaluate` `{dataset_name, text_columns, label_column, …}` →
  background job; result stored under `metadata.evaluations[]` (never overwrites the
  training metrics); UI: "Evaluate on…" in the model panel with a comparison table
  across models evaluated on the same dataset.
- **Why:** the only honest way to say "model B beats model A" is the same holdout;
  today that is a script (`scripts/eval_holdout.py`) against a running server.
- Files: `app/evaluate.py` (new; reuses `load_dataset`, `prepare_targets`,
  `compute_metrics`), routes, UI.
- Deviation on the test: "reproduces the bundle's `per_label_f1` within 1e-6 for the
  holdout rows" is not achievable without replaying the training split (seed, sizes),
  which the endpoint deliberately does not do — it scores the rows it is given. What is
  pinned instead is what makes the number honest: truth aligned to the MODEL's label
  space, unknown labels reported, uncoverable rows excluded and counted, `metrics: null`
  when nothing comparable remains, and the decision rule serving actually applies.
- `prepare_targets` is NOT reused: it binarizes over the dataset's own label set, which
  is a different width from the model's output. Alignment to `model.classes` is the
  whole point.
- Deviation on the UI: no cross-model comparison TABLE. The bias that makes a comparison
  valid is "same dataset", and the model panel already shows each run's dataset beside
  its scores — a table joining models would need every bundle's metadata on opening one
  dialog. Worth building when two models have actually been scored on one dataset.

Effort: 10–12 person-days.

---

## Phase 3 — Recognition rate and speed, measured (4–8 weeks, parallelisable)

Every item ships as a benchmark script first; adoption needs the stated gate.

| # | Item | Expected effect | Gate to adopt | Effort |
|---|---|---|---|---|
| ⚠️ A2 **re-priced by A3 — recommend dropping** | **Warm-started C path** during *selection only*: per label, fit C ascending with `warm_start=True`, keep the deploy fit on the plain `OneVsRestClassifier` so bundles stay pure sklearn. | A2 harvests the same pool A3 measured: solver iterations in the C search. A3 halved them for a median 4.4 % of the phase, so the whole pool is a small share of it — an estimate chained across two measurements, not a measurement | Unchanged (≥ 15 % wall-clock, identical `best_C` and OOF F1 ±1e-4) — but the arithmetic says 15 % is out of reach at this shape. **Owner decision needed:** drop, or measure the fit share directly first | M, 2–3 days |
| 🔴 A3 **measured 2026-09-09, rejected** | Looser `tol` (1e-3) for selection fits, 1e-4 for deploy. Mechanism shipped as `Profile.selection_tol`, off everywhere; `scripts/benchmark_selection_tol.py` re-measures it | Harmless (same `best_C`, ΔF1 −0.000029) but **median 4.4 % of the selection phase over four runs** (−2.7 / +3.4 / +5.4 / +11.2) | ❌ 15 % not reached. The solver tail is real — 7.02 → 3.69 newton-cg iterations, one fit 26 s → 12 s — the phase is just mostly vectorization, scoring and threshold search | S, done |
| B3 | **Iterative stratification** for the holdout split and the K folds (multilabel-aware; ~60 lines, no dependency) | Rare labels get positives in every fold → stabler per-label thresholds and metrics | Macro F1 not worse; per-label F1 variance across seeds lower | S–M, 1–2 days |
| B4 | **Threshold shrinkage** for labels with few validation positives toward the global threshold | Fewer degenerate thresholds on the tail | Macro F1 ≥ baseline on both targets | S, 1 day |
| B5 | **Per-label calibration** (isotonic on OOF probabilities) so `confidence` reads as a probability; `class_weight="balanced"` inflates positives today | Better `confidence`/`baseline_diff` semantics for the UI; decisions unchanged | Brier/ECE improve; F1 unchanged | S–M, 1–2 days |
| B2 ✅ **2026-09-09** (merge tool excepted — it belongs to the data-prep app, not here) | **Feedback loop**: `POST /feedback` (text, model, predicted, corrected, source) → JSONL; `GET /feedback/export` → training-compatible CSV; UI "correct this" on results; ~~merge tool into the dataset workflow~~ | The recognition rate improves with use instead of only with re-exports | Manual: a 200-row correction set retrains to a higher F1 on a fixed holdout | M, 3 days |
| B6 | **Label hierarchy**: persist SKOS `broader` in `label_names.json` (fetch script) and in the bundle; `/predict` can return the broader concept, UI groups by parent | Fewer "wrong sibling" errors visible to editors; hierarchy-consistent output | Owner review on 50 predictions | M, 2–3 days |
| ✅ D6 **done 2026-09-09** | **DE/EN UI**: `strings-de.js` / `strings-en.js` (strict JSON in a one-line assignment, loaded as scripts), toggle in the top bar and on the login bar, language resolved as stored choice → browser preference → German; `docs/ui-guide.md` relabelled to the German UI | The audience reads German | ✅ 331 keys, both maps identical; `tests/test_ui_i18n.py` (13 tests) fails on a missing key, an orphan key, a template-built key, or prose left in `index.html` (parsed); the JS side is netted by three verified patterns | M, done |
| B7 | Train school subjects on the combined 426 k export with `best` (~1.7 h) and evaluate against `faecher_300k_auto` via B1 | Data is the biggest lever left | B1 comparison on the same holdout | S (compute) |

**Owner decisions, 2026-09-09.** Training: **short benchmark runs allowed** (≈30 k rows,
minutes each) — this unblocks P1, A2, A3, B3, B4 and B5. Still held: **B7** (426 k, ~1.7 h)
and the retrain of `bildungsstufe_ai_cv5`. **D6 approved**, both languages switchable.

Effort: ~15 person-days spread over the period; each item independent.

---

## Owner proposals, 2026-09-09 — checked against the code, not yet scheduled

Five suggestions arrived after D6, aimed at the cheapest levers: how C and the
decision thresholds are chosen, and how the training text is assembled. Nothing is
scheduled yet; what follows is what each one is actually worth once the claim behind
it was checked against this repository. Where a claim was measurable, it was measured
rather than argued.

| # | Proposal | Status after checking | What it would cost |
|---|---|---|---|
| **C1** | **Pick C on tuned thresholds, not on a fixed 0.5.** Today `tuning.py` scores every candidate with `_default_decision` (a flat 0.5 cut for multilabel) and tunes thresholds only on the winner, so a candidate that would be better *with its own thresholds* can be eliminated before it is ever tried. | 🟢 **Confirmed, and the better procedure already exists in this repo**: `scripts/benchmark_field_weights.py` tunes thresholds per C and *then* picks the best. The production pipeline never adopted it. Multilabel only — single-label serving is argmax and reads no threshold. | No new fits: in CV mode the OOF probabilities for every C are already in memory (`oof[c]`). Threshold tuning per candidate is \|grid\| × the current tuning cost. |
| **C2** | **Stop searching thresholds in 0.05 steps.** `_DEFAULT_GRID` is 0.05…0.95; anything between two steps, or outside the range, is unreachable. Derive candidates from the observed scores (a PR curve), and shrink toward the global threshold where a label has few positives. | 🟢 **Confirmed** for the grid. The shrinkage half is plan item **B4**, already scheduled — C2 and B4 should be one piece of work, not two. | No new fits. Sorting each label's scores is O(n log n) per label on data already held. |
| **C3** | **Apply the field weighting consistently at train and predict time.** | 🔴 **Already done** — the claim is out of date. `text_column_weights` is a validated `TrainRequest` field (`schemas.py:73`, rejects columns the request does not train on), `prepare.py:128` honours it over the config default, the admin UI always sends it, and `/predict/csv` reads the weights back out of the model's own metadata (`routes/predict_bulk.py:60`) so a CSV is assembled the way the model was fit. What remains is inherent: `/predict` with a bare text cannot know how the caller assembled it, which is why the training form says so. | — |
| **C4** | **Merge labels across duplicate texts** instead of keeping the first row and discarding the rest (`data.py:264`). | 🟡 **Mechanism confirmed, effect measured as negligible.** On `data_30k.csv` dedup drops **7 445 of 32 516 rows (22.9 %)** across 5 406 duplicate groups — but only **2 groups** contain a label the kept row lacks, i.e. **2 lost label assignments in total**, and 1 genuine disagreement. `data_30k_ai.csv` and `data_30k_base.csv` have no duplicate texts at all. Duplicates here are the same item's metadata repeated, so they carry the same labels. | Measured 2026-09-09; the review effort would find almost nothing **on these three exports**. A differently shaped export (one row per collection membership) could differ — re-measure before dismissing it there. |
| **C5** | **Two small experiments**: `class_weight="balanced"` against unweighted (with thresholds retuned either way), and a relative weight between the word and character TF-IDF blocks. | 🟢 **Confirmed as unmeasured.** `classifier.py:33` hardcodes `class_weight="balanced"`; `vectorizers.py:83` `hstack`es the two blocks with no scaling. Both interact with **B5** (calibration) — `balanced` is precisely what makes `confidence` read high. | Comparison runs only; neither changes the bundle size. |

### C1 in detail — the package to build first

Four steps, each its own commit, because the first one is a refactor and bundling a
refactor into a feature is how a bisect stops being useful.

0. **Split `tune_thresholds` into a column core and a URI-keyed wrapper.** The maths is
   about columns; the `uri -> threshold` dict is presentation at the edge. Without this,
   scoring a candidate by its own thresholds means pushing `classes` and `per_label`
   through `select_c`, which already carries nine parameters. Behaviour-preserving, and
   it lets `apply_thresholds` become one vectorised comparison instead of a Python loop
   over labels.
1. **`Profile.select_c_on_tuned_thresholds: bool = False`** — off until the gate is met,
   the same shape as `refit_vectorizer_per_fold` and `selection_tol`.
2. **CV path**: tune thresholds on `oof[c]` for every candidate, pick the (C, thresholds)
   pair together, and keep the winner's thresholds instead of re-deriving them.
3. **Holdout path**: `select_c` scores each candidate with its own thresholds and returns
   them. This also removes a redundant `head.predict_proba(x_va)` that `deploy.py:108`
   runs today on probabilities `select_c` had already computed and discarded.

**Gate.** Macro F1 up by >= 0.002 on the 26 k target, *and* `predicted_labels_per_row`
not more than 10 % above the baseline. The second half is the owner's own condition and
the reason `compute_metrics` already records that pair: a threshold rule can always buy
macro F1 by asserting more labels per row, and that is a different product rather than a
better model. Precision and recall are reported beside them so the trade is visible.

**Cost.** `tune_thresholds` runs |c_grid| times instead of once — no additional model
fits, which is what makes this the cheapest quality lever left in the plan.

**Sequencing, if these are taken up:** C1 and C2 are one experiment, not two — both
change how a decision threshold is chosen, and measuring them apart would attribute the
same gain twice. They fold naturally into **B4**, which already owns the shrinkage half.
Run them on a fixed holdout with the label set unchanged, and report **precision, recall
and `predicted_labels_per_row` beside macro/micro F1**: a threshold change that buys F1
by asserting more labels per row is a different product, not a better model — which is
exactly why `compute_metrics` already records that pair.

---

## What is deliberately *not* in the plan

- **New backends** (embeddings, SVM): measured in July; TF-IDF+LogReg wins on this data.
- **Multi-worker / external queue / database:** the single-worker design is a feature
  (one PVC, no shared state); the Phase-2 queue stays in-process and persisted to disk.
- **CV10 anywhere:** measured knee at 5 folds; stays request-only.
- **A JS framework or build step:** the vanilla UI is ~1,650 lines of code plus ~660
  lines of string map, CSP-clean apart from F1; splitting into per-tab files keeps it
  that way. D6 added i18n without one — two script tags and a `t()`, no bundler.

## Verification plan (applies to every phase)

- Gates per PR: `ruff` · `mypy` · `pytest` (the count in README updated in the same PR)
  · OpenAPI smoke · for UI items a browser pass with **zero CSP console errors** and a
  keyboard walk-through.
- Benchmarks produce a JSON file next to the existing `*_results.json` and a table in
  README or `docs/model-approach-comparison.md`; the number, not the expectation, is
  what gets written down.
- Regression: the full suite before merge; a real-data smoke (`data_30k_ai.csv`,
  `auto`) after any change to `tuning.py`/`deploy.py`.

## Decisions needed from the owner

1. Approve Phase 0 now (commit slices + F1 + O3 cleanup, incl. deleting the five
   experimental/backup bundles listed in 0.3).
2. `data/label_names.json`: commit as a dated snapshot, or keep it out of git?
3. Phase-1 default for P1 once measured: shared vectorizer on by default if the gate
   passes?
4. Order of Phase 2: D2 (batch classification) first, or A5/D5 (queue + history)?
5. Phase 3 items to drop: B5 (calibration) and B6 (hierarchy) are the two with the
   least certain payoff.
