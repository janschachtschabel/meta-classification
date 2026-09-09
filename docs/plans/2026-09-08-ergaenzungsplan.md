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

### C1 · Bundle manifest + model card (S–M, 1–2 days)
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

### C2 · Share-link management (S, ½ day)
- `GET /share` (admin: id, kind, name, expires_at, downloads) and `DELETE /share/{id}`
  (admin); optional `max_downloads` on create. UI: "Active share links" box on the
  Models and Datasets tabs with Revoke.
- Files: `app/sharing.py` (+`list`, `revoke`, `count_download`), `app/routes/models.py`,
  `app/static/ui/manage.js`. Tests: create → list → revoke → `GET /share/{id}` 404.

### C3 · Streamed transfers (S, ½ day) — P2
- `export_zip` writes to a `tempfile` and returns a `FileResponse`; dataset uploads
  spool to `<name>.part` and `os.replace`. Tests: a crash between write and replace
  leaves no listed dataset.

### B8 · Per-label diagnostics endpoint (S, ½ day)
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

### P1 · Shared-vectorizer cross-validation (S–M, 1 day incl. benchmark)
- `cross_val_evaluate` accepts a prefitted matrix: `fit_transform` once over all rows,
  slice per fold, reuse for the deploy fit. Profile flag `refit_vectorizer_per_fold`
  (default decided by the benchmark). `scripts/benchmark_shared_vectorizer.py` reports
  macro/micro F1 and wall-clock for both modes on `data_30k_ai.csv` and the
  university target.
- **Gate:** adopt as default only if |Δ macro F1| < 0.002 on both; expected saving
  ≈ 17 % (`auto`) / 20 % (`best`) at 156 k rows (README: 2.3 min per pass).
- Files: `app/tuning.py`, `app/deploy.py`, `app/profiles.py`, `config.yaml`, README
  (measured table). Tests: both modes produce identical OOF shapes; the flag is
  recorded in `metrics.json`.

### A1 · Split `data.py` (S, 1 h)
- Move `_rejoin_split_names`, `_pair_names` and `label_vocabulary` to
  `app/label_names.py`; behaviour-preserving, tests move with them.
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

## Phase 2 — The UI catches up with the API (2–4 weeks)

### D2 · Batch classification (M, 2–3 days)
- Query tab: "Many texts" mode — paste one text per line *or* upload a CSV, choose
  the text column(s) and the weights the model expects (read from the bundle), run
  in chunks of 1000 through `/predict/batch`, show a table, download the result as
  CSV (`uri`, `label`, `confidence`, `above_threshold` per row). Server: a
  `POST /predict/csv` endpoint that takes an uploaded CSV and streams a CSV back
  (bounded by `max_upload_mb`; chunked, no full-file materialisation).
- **Why:** the daily editorial job is "classify these 500 new items", not one text.
- Files: `app/routes/predict.py` (+ `app/predict_csv.py`, new), `app/static/ui/query.js`
  (split from `app.js`). Tests: 3-row CSV round-trip; text weights applied as the
  bundle records them; oversize → 413.

### D3 · Dataset inspector + training pre-flight (M, 2 days)
- Datasets tab: click a row → columns, first rows (`GET /datasets/{name}`), and an
  "Analyze" button (`/datasets/analyze`) showing label counts, the `labels_with_N+`
  table and warnings. Training tab: after picking dataset + label field, show the
  same numbers inline with a recommended `min_samples_per_label` and an estimated
  duration from rows × profile (README cost model, labelled "estimate").
- Files: `app/static/ui/datasets.js`, `training.js`, `app/dataset_stats.py` (+`support`
  per label). Tests: analyze output includes the fields the UI reads.

### D4 · Explain view (S–M, 1 day)
- "Why?" on a prediction → `/predict/explain`; the top words per label rendered as
  chips with their impact; `all_scores` collapsible.

### A5 · Server-side training queue + D5 persisted history (M, 3 days)
- `POST /train` while busy → 202 with a queue position (bounded queue, e.g. 10);
  jobs persist to `models_dir/.jobs.jsonl` (request, status, metrics, duration);
  `GET /train/history`; the UI's tab-bound queue goes away (the browser can close).
  `TrainingJob` becomes `JobRunner` with `kind` (training | evaluation) so B1 reuses it.
- **Why:** "train five label fields overnight" currently needs an open tab; history
  is the only way to compare runs without opening bundles.
- Files: `app/jobs.py` (split: `jobs.py` runner + `job_history.py`), routes, UI.
  Tests: queue order, persistence across a fresh `JobRunner`, hard stop clears the
  queue.

### B1 · Evaluate an existing model on a dataset (M, 2 days)
- `POST /models/{name}/evaluate` `{dataset_name, text_columns, label_column, …}` →
  background job; result stored under `metadata.evaluations[]` (never overwrites the
  training metrics); UI: "Evaluate on…" in the model panel with a comparison table
  across models evaluated on the same dataset.
- **Why:** the only honest way to say "model B beats model A" is the same holdout;
  today that is a script (`scripts/eval_holdout.py`) against a running server.
- Files: `app/evaluate.py` (new; reuses `load_dataset`, `prepare_targets`,
  `compute_metrics`), routes, UI. Tests: evaluating the fixture model on its own
  training CSV reproduces the bundle's `per_label_f1` within 1e-6 for the holdout
  rows.

Effort: 10–12 person-days.

---

## Phase 3 — Recognition rate and speed, measured (4–8 weeks, parallelisable)

Every item ships as a benchmark script first; adoption needs the stated gate.

| # | Item | Expected effect | Gate to adopt | Effort |
|---|---|---|---|---|
| A2 | **Warm-started C path** during *selection only*: per label, fit C ascending with `warm_start=True` (newton-cg supports it), keep the deploy fit on the plain `OneVsRestClassifier` so bundles stay pure sklearn. | Fewer solver iterations on candidates 2..n; ⚪ theoretical 20–40 % of search time | ≥ 15 % wall-clock on `auto` at 30 k rows, identical `best_C` and OOF F1 (±1e-4) | M, 2–3 days |
| A3 | Looser `tol` (1e-3) for selection fits, 1e-4 for deploy | ⚪ small | same as A2 | S, ½ day |
| B3 | **Iterative stratification** for the holdout split and the K folds (multilabel-aware; ~60 lines, no dependency) | Rare labels get positives in every fold → stabler per-label thresholds and metrics | Macro F1 not worse; per-label F1 variance across seeds lower | S–M, 1–2 days |
| B4 | **Threshold shrinkage** for labels with few validation positives toward the global threshold | Fewer degenerate thresholds on the tail | Macro F1 ≥ baseline on both targets | S, 1 day |
| B5 | **Per-label calibration** (isotonic on OOF probabilities) so `confidence` reads as a probability; `class_weight="balanced"` inflates positives today | Better `confidence`/`baseline_diff` semantics for the UI; decisions unchanged | Brier/ECE improve; F1 unchanged | S–M, 1–2 days |
| B2 | **Feedback loop**: `POST /feedback` (text, model, predicted, corrected, source) → JSONL; `GET /feedback/export` → training-compatible CSV; UI "correct this" on results; merge tool into the dataset workflow | The recognition rate improves with use instead of only with re-exports | Manual: a 200-row correction set retrains to a higher F1 on a fixed holdout | M, 3 days |
| B6 | **Label hierarchy**: persist SKOS `broader` in `label_names.json` (fetch script) and in the bundle; `/predict` can return the broader concept, UI groups by parent | Fewer "wrong sibling" errors visible to editors; hierarchy-consistent output | Owner review on 50 predictions | M, 2–3 days |
| D6 | **DE/EN UI** via a JSON string map, German default, toggle in the top bar; `docs/ui-guide.md` stays the German manual | The audience reads German | Every string in the map; no hardcoded text (grep test) | M, 2 days |
| B7 | Train school subjects on the combined 426 k export with `best` (~1.7 h) and evaluate against `faecher_300k_auto` via B1 | Data is the biggest lever left | B1 comparison on the same holdout | S (compute) |

Effort: ~15 person-days spread over the period; each item independent.

---

## What is deliberately *not* in the plan

- **New backends** (embeddings, SVM): measured in July; TF-IDF+LogReg wins on this data.
- **Multi-worker / external queue / database:** the single-worker design is a feature
  (one PVC, no shared state); the Phase-2 queue stays in-process and persisted to disk.
- **CV10 anywhere:** measured knee at 5 folds; stays request-only.
- **A JS framework or build step:** the vanilla UI is 1,240 lines and CSP-clean apart
  from F1; splitting into per-tab files keeps it that way.

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
