# MetaClassify

**Torch-free, CPU-only text-classification API — train on your metadata, serve multiple models via REST.** No GPU, low RAM, running equally well on **Windows and Linux**.

It covers all features of the previous version: **training** on your own CSV metadata, **classification** via the API, **automatic metrics**, **sharing** models and **switching** between models for different classification tasks.

## Why MetaClassify

- **Modern & lean:** TF-IDF (word + character n-grams, sparse) + scikit-learn LogisticRegression. **Torch-free, no model download, no GPU.** (Embeddings were tested extensively but did not beat TF-IDF on this metadata — see below.)
- **Automatic & self-applying:** the `auto` profile selects `C` on the validation split and tunes thresholds (global + per-label). These are **stored in the model bundle and applied automatically at `/predict`** — by default the tuned thresholds decide which labels are returned (an explicit `top_k` switches to a ranking of the N most probable labels, each flagged `above_threshold`). Task type (binary/multiclass/multilabel) is auto-detected; `min_samples_per_label` is auto-scaled (and can also be set per request).
- **Data preparation:** HTML tags, Markdown markup and HTML entities are stripped, whitespace is normalized (`clean_text`).
- **Honest metrics:** train/val/**test** split by default — tuning on validation, reported metrics on the untouched test split. Or use **k-fold cross-validation** (`split.cv_folds` in `config.yaml`, or per request via the `cv_folds` body field, e.g. 5): every row is used for both training and validation (out-of-fold metrics) and the deployed model is fit on **100 %** of the data. *Caveat:* with CV, `C` and the thresholds are selected on the same out-of-fold predictions the metrics report — a mild optimism (no row is ever scored by a model that saw it, but the two tuning choices are in-sample). The classic split keeps strict separation.
- **Secure:** pickle-free models (**skops**), import via file upload only (no server-side URL fetch → no SSRF), API-key auth (constant-time), CORS allowlist, upload limit.
- **Low RAM & scalable:** sparse features + a linear head; **LRU model cache**; resources tunable via `APIV3_N_JOBS` (cores) and `APIV3_TFIDF_MAX_*_FEATURES` (RAM).

## Installation

Requires **Python ≥ 3.11**.

```bash
# from the repo root
python -m venv .venv && . .venv/Scripts/activate   # Windows
# source .venv/bin/activate                         # Linux/Mac
pip install -r requirements.txt -c requirements.lock   # lock = the tested versions
cp .env.example .env   # set your API keys (see docs/configuration.md for all options)
```

## Start

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
# Admin UI:   http://127.0.0.1:8000/ui/
```

**Admin UI** (`/ui/`): a self-contained static page (vanilla JS, no build step, no external assets) for datasets (upload, download, share links, delete), training with live progress (several label fields at once → the UI queues one training per field sequentially and derives the model names as `base_field`), model management (metrics, download, share links, import, delete) and test queries. Sign in with an API key — it is kept in `sessionStorage` (gone when the tab closes) and sent as `X-API-Key` on every request, so the page needs no extra auth mechanism; readonly keys can query, management actions need the admin key. The UI mirrors the server's auth setting exactly like `/docs`: with `APIV3_AUTH_ENABLED=false` it skips the sign-in entirely (badge "auth disabled — full access"). Every tab carries a built-in plain-language help section ("What do the results mean?" etc.); a full non-technical walkthrough for editorial users is in [`docs/ui-guide.md`](docs/ui-guide.md) (German, matching its audience). Disable the UI with `APIV3_UI_ENABLED=false`.

All endpoints require the `X-API-Key` header with one of two keys (roles): the **admin key** (`APIV3_API_KEY_ADMIN`) guards the critical/expensive endpoints — `/train`, `/train/stop` and all dataset/model management (import, export, delete, analyze) — while the **readonly key** (`APIV3_API_KEY_READONLY`) suffices for classification and status (`/predict*`, `/train/status`, listings). Exceptions needing no key: `/health`, `/metrics` (operational gauges only) and `GET /share/{id}` — a share link is a **bearer capability** (the unguessable id plus its expiry are the authorization, so it can be handed to someone without a key; creating links stays admin-only). For purely local use set `APIV3_AUTH_ENABLED=false` in `.env`.

## Workflow

1. Put a CSV into `data/` (or upload via `POST /datasets/import`). **`.csv.gz` works everywhere `.csv` does** — listing, inspection, download and training — and is the practical choice for large exports: the WLO full exports are 126–195 MB compressed against ~1.4 GB plain.
2. `POST /train` with `text_columns`, `label_column`, a profile (`auto`/`fast`) — optionally `text_column_weights` to emphasise short, dense fields.
3. `GET /train/status` until `completed` (includes test metrics) — or `error` with a plain-text reason (e.g. no label reaching `min_samples_per_label`).
4. `POST /predict` with `model_name`.
5. Multiple tasks = multiple models: select per request via `model_name`.

### Quick example (curl)

```bash
# Train (admin key) -> poll status -> predict (any valid key).
curl -X POST localhost:8000/train -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" -d '{
  "dataset_name": "data_30k.csv", "model_name": "subjects",
  "text_columns": ["properties.cclom:title","properties.cclom:general_description","properties.cclom:general_keyword"],
  "label_column": "properties.ccm:taxonid", "optimize_parameters": "auto",
  "label_filter": "http://w3id.org/openeduhub/vocabs/discipline/",
  "min_samples_per_label": 20,
  "text_column_weights": {"properties.cclom:title": 2, "properties.cclom:general_keyword": 2}
}'
curl localhost:8000/train/status -H "X-API-Key: $RO_KEY"
curl -X POST localhost:8000/predict -H "X-API-Key: $RO_KEY" -H "Content-Type: application/json" \
  -d '{"texts": ["Bruchrechnung und Gleichungen lösen"], "model_name": "subjects",
       "include_label_f1": true}'
```

With `APIV3_AUTH_ENABLED=false` the `X-API-Key` header is optional.

### Weighting text fields (`text_column_weights`)

Title and keywords carry far more signal per word than a long description, but a
long description drowns them out: TF-IDF counts terms, and the description simply
supplies more of them. `text_column_weights` repeats a column when the training
text is assembled, which is what restores its term frequency:

```json
"text_columns": ["title", "description", "keywords"],
"text_column_weights": {"title": 2, "keywords": 2}
```

**This is on by default.** `config.yaml` ships
`preprocessing.text_column_weights` with title + keywords at 2× (the measured optimum
below), applied whenever a `/train` request omits the field and narrowed to the columns
that request actually trains on — so the default is harmless for a CSV with different
column names. Send your own mapping to override it, or `{}` to train unweighted;
`GET /train/profiles` reports the active default and the admin UI pre-fills from it.

- Values are `1…10`; keys must be among `text_columns` (a typo is a `422`, not a silent no-op).
- `sublinear_tf` damps repetition logarithmically, so `2` is worth ~1.7×, not 2×.
- **Training-time only, and that has a consequence:** `/predict` takes one opaque
  string, so the API cannot re-apply the weights for you. A model trained with
  weights expects input built the same way — **assemble the text you send to
  `/predict` with the same repetitions**, otherwise its tuned thresholds sit on a
  slightly different feature distribution than they were tuned on. The weights are
  recorded in the bundle (`text_column_weights` in `GET /models/{name}`), so a
  client can always look up what a given model expects.

🟢 **Measured** on `data_30k_ai.csv` (48 subjects, identical rows/split/C grid per
variant — only the text assembly differs; `scripts/benchmark_field_weights.py`):

| Variant | F1 macro | Δ macro | F1 micro | Fit time |
|---------|---------:|--------:|---------:|---------:|
| baseline (everything 1×) | 0.7063 | ±0 | 0.7937 | 36.7 s |
| **title 2× + keyword 2×** | **0.7208** | **+0.0145** | **0.7989** | 34.9 s |
| title 3× + keyword 3× | 0.7179 | +0.0116 | 0.7974 | 32.4 s |
| keyword 2× only | 0.7208 | +0.0145 | 0.7966 | 34.6 s |
| title 2× only | 0.7015 | **−0.0049** | 0.7902 | 33.7 s |

What that says, and it is not what one would guess:

- **`{title: 2, keyword: 2}` is the recommended setting** — best or tied on both
  metrics, and the largest single quality gain measured on this data.
- **The keyword column carries the gain.** Boosting keywords alone already yields
  the full +0.0145 macro; boosting the *title* alone lands **below** baseline.
  Title 2× only earns its place next to a keyword boost (best micro), never on its own.
- **More is not better:** 3× is worse than 2×. Don't push the multiplier up.
- **It is free.** The repeated column adds almost no non-zeros (a repeat raises term
  counts, it does not add new terms), so matrix size, RAM and fit time are unchanged.
  The only real cost is the train/serve consistency requirement above.

### Label display names (`data/label_names.json`)

Predictions carry a human-readable `label` next to the URI, taken from the CSV's
`<label_column>_DISPLAYNAME` column. That column is a trap worth knowing about: it uses the
**same separator as the URI list**, so a name that itself contains that character
(`"Rechts-, Wirtschafts- und Sozialwissenschaften"`) splits in two and shifts every later
name onto the wrong URI. 🟢 Measured on `data_300k.csv`: 6.09 % of rows, which had given
34 of 119 higher-education labels the name of a **different subject**.

Names are therefore only accepted when the counts provably line up, plus a reconstruction
pass that uses the URI count as the arity constraint (🟢 agreed with clean rows 75/75).
**A row that cannot be reconciled contributes no name at all** — `/predict` then shows the
URI, which is honest; a confident wrong subject is not.

For complete names, supply the vocabulary itself:

```bash
python scripts/fetch_vocab_labels.py    # SKOS -> data/label_names.json (build-time only)
```

Training prefers that file over anything derived from the CSV. Already-trained bundles can
be repaired **without retraining** — `uri_to_label` is presentation-only JSON inside
`config.json`:

```bash
python scripts/patch_bundle_labels.py --apply
```

See [`docs/configuration.md`](docs/configuration.md#datalabel_namesjson--authoritative-label-display-names-optional) for the details.

### Model documentation (`PUT /models/{name}/info`)

A bundle records what the pipeline *measured* — dataset file name, columns, profile,
`C`, folds, rows, metrics, training time. It cannot know what only the person training
it knows, and that is exactly what a recipient of a shared model needs:

| Field | For |
|---|---|
| `author` | who trained it, and how to reach them |
| `description` | what it is for — and what it is **not** for |
| `data_source` | where the data came from (`dataset` is only a file name, meaningless elsewhere) |
| `license` | terms, for models that leave the house |

All optional and length-bounded, stored inside `metrics.json` so they travel in the
exported ZIP, and settable either at training time (`info` in the `/train` body) or
afterwards — documentation is presentation-only, so fixing a typo costs no retrain.
The admin UI has an **Info** button per model.

**The label vocabulary is not among them**, on purpose: it is derived from the class
URIs and reported as `label_vocabulary` by `GET /models/{name}`. A typed-in value can
be wrong; a derived one cannot. 🟢 Checked against the 14 bundles of the local model
store — every one resolved to a single namespace (`…/vocabs/discipline/`,
`…/vocabs/educationalContext/`, `…/vocabs/hochschulfaechersystematik/`), and a model
mixing two vocabularies correctly reports none.

### Container labels are dropped, not learned

A label value ending in `/` names a namespace rather than a concept and never becomes a
class. 🟢 Measured on `data_300k.csv`: the bare vocabulary root `…/vocabs/discipline/` was
attached to 522 rows and had trained as an ordinary label scoring **F1 0.4096** — a class
`/predict` could return that means nothing. `min_samples_per_label` cannot catch it (522 rows
clears any threshold), so the guard is structural. It stays narrow on purpose: a genuine
broader concept has an id and is kept, because the label hierarchy is real signal.

Older bundles still carry such a class — loading one logs a warning, and
`scripts/prune_bundle_labels.py --apply` removes it without retraining (it drops the matching
estimator too and verifies the remaining probabilities are bit-identical).

## Profiles (`config.yaml`)

Three rungs, strictly ordered by cost — the name is a truthful price tag:

| Profile | Char n-grams | `C` grid | Evaluation | Head fits¹ | Trains the shipped model on |
|---------|:------------:|----------|------------|-----------:|-----------------------------|
| `fast` | no | `[2, 32]` | holdout split | 2.4 | 85 % of rows |
| `auto` | (5,5) | `[2, 8, 32]` | 3-fold CV | 7 | **100 %** *(recommended)* |
| `best` | (5,5) | `[2, 8, 32]` | 5-fold CV | 13 | **100 %** |

¹ In units of the full dataset: `(folds − 1) × |C_grid| + 1`, the term that dominates
wall-clock. `best` therefore costs ~1.9× `auto`; `fast` runs in seconds but is for
iteration — it is the only profile that does **not** deploy a model trained on all your
data.

**The `C` range is fixed at 2 … 32 for every profile, and that is measured, not assumed.**
Two facts pin it:

- **The lower and upper ends are both needed.** This project's targets optimize at
  *opposite* ends — the subject model picks `C=32`, the educational-level model picked
  `C=2` — so a two-candidate `[4, 16]` would reach neither. Pinned by
  `test_every_profile_brackets_both_known_optima`.
- **Above 32 quality drops.** 🟢 Measured on a holdout split for both targets
  (`scripts/benchmark_c_range.py`, zero convergence warnings):

  | C | 8 | **32** | 128 | 512 | 2048 |
  |---|---:|---:|---:|---:|---:|
  | school, macro F1 | 0.7504 | **0.7531** | 0.7511 | 0.7493 | 0.7479 |
  | university, macro F1 | 0.7803 | **0.7814** | 0.7812 | 0.7785 | 0.7785 |

  Reaching further costs quality *and* ~50 % more fit time, so
  `test_no_c_grid_reaches_past_the_measured_useful_range` forbids it.

Within that range, resolution is cheap to give up: a 3-candidate grid at 4× steps selects
the same `C` at identical F1 as an 8-candidate grid at 2× steps. So `auto` and `best` share
the same grid, and **the only thing `best` buys is the fold count** — see below.

> An earlier revision had `best` search `[0.5 … 128]`, on the theory that four consecutive
> runs picking the grid maximum meant the optimum lay beyond it. The table above refutes
> that: those were ties on a flat plateau. At a fixed 5-fold CV, `C=32` and `C=128` scored
> 0.8135 vs 0.8130. Every bundle records the grid it searched (`c_grid`) next to `best_C`,
> so an edge pick stays visible — just read it as a tie, not as a missing optimum.

**The fold count is the whole difference between `auto` and `best` — and 5 is the knee.**
Fold count does not decide whether the evaluation is honest: under any `k ≥ 2` every row is
scored by a model that never saw it, and the deployed model is refit on 100 % of the rows.
`k` only sets how much data the *evaluation* models train on, so a smaller `k` makes the
reported score slightly **pessimistic**, not inflated. 🟢 Isolated on the university target
at a fixed `C=32`:

| folds | evaluation models see | macro F1 | gain | run time |
|---|---:|---:|---:|---:|
| 3 (`auto`) | 67 % | 0.7992 | — | — |
| **5 (`best`)** | 80 % | **0.8135** | **+0.0143** | 9.1 min |
| 10 | 90 % | 0.8166 | +0.0031 | 16.7 min |

The returns collapse after 5 while the cost keeps doubling, so **no shipped profile uses
`cv_folds: 10`** — it is available per request if you want it, but it is not worth 2× the
time for three thousandths.

Add your own profiles in `config.yaml` (fields: `C_grid`, `cv_folds`, `tune_threshold`, `threshold_per_label`, `use_char`, `max_word_features`, `max_char_features`).

## Endpoints (excerpt)

- **Training:** `POST /train`, `GET /train/status`, `POST /train/stop`, `GET /train/profiles`
- **Classification:** `POST /predict`, `POST /predict/batch`, `POST /predict/multi` (several models = target fields in one call; each model applies its own tuned thresholds, evaluation stays per model), `POST /predict/explain`. All predict endpoints can attach two reliability signals per prediction (both always on in `/predict/explain`):
  - `baseline_diff` (`include_baseline_diff=true`) — confidence minus the model's empty-text prediction. A high confidence with a diff near zero means the label fires for almost anything, not for this text.
  - `label_f1` (`include_label_f1=true`) — this label's F1 from the training evaluation. Confidence says how sure the model is *here*, `label_f1` how much that is worth: `Politik 0.95` on a label scoring 0.68 deserves a human look, `Mathematik 0.95` on a label scoring 0.95 does not. `null` for labels the bundle has no score for.
- **Models:** `GET /models`, `GET /models/{name}`, `GET /models/{name}/labels` (per-label F1, support and threshold, weakest first), `PUT /models/{name}/info`, `DELETE /models/{name}`, `POST /models/{name}/export`, `POST /models/import`
- **Share links:** `GET /share/{id}` (public bearer download), `GET /share` and `DELETE /share/{id}` (admin: review what is outstanding, withdraw it early)
- **Datasets:** `GET /datasets`, `GET /datasets/{name}`, `POST /datasets/analyze`, `POST /datasets/{name}/validate`, import/export/delete
- **System:** `GET /health`, `GET /config`, `GET /metrics` (Prometheus text format, public — operational gauges only: uptime, model counts, training status)

## Status & monitoring

`GET /train/status` returns live: `status` (idle/running/completed/error/stopped), `phase` (loading → preparing → features → selecting → threshold → evaluating → saving → done), `phase_detail` (e.g. `Testing regularization: C=8 (5/6) – best F1 so far 0.629`, or `Fold 3/5: C=2.0 (14/30 fits)` in CV mode), `progress` (0–100), `message`, `elapsed_seconds`, `eta_seconds` (estimated time remaining), `seconds_since_heartbeat` (age of the newest progress signal while running — it keeps growing when the training thread stalls silently, while `elapsed_seconds` grows either way), `model_name`, `results` (metrics on completion), `error` (failure text, set alongside `status: "error"`). **Progress moves within the long phases** — the C search spreads over 55→75 % and a cross-validation run over 45→90 % (one step per head fit), so the ETA stays meaningful instead of freezing at a phase boundary. Cancel via `POST /train/stop` (`hard=false`/`true`); it takes effect promptly (checked between the C fits and before deployment). The admin UI shows the same status as a live progress card on the Training tab plus a compact chip in the top bar that stays visible on every tab while a run is active.

For Prometheus, `GET /metrics` exposes operational gauges (uptime, models on disk / in the LRU cache, training running + progress) in the text exposition format — hand-rolled, no client-library dependency. The Helm chart's ServiceMonitor (`global.metrics.servicemonitor.enabled`) scrapes it.

## Resources & tuning

**Fast ↔ good** via the profile (`optimize_parameters`): `fast` (word-only, holdout split, 2 `C` values → seconds, for iteration), `auto` (word + char, 3-fold CV, 3 `C` values — recommended), `best` (word + char, 5-fold CV, the same 3 `C` values — the fold count is the whole difference, ~1.9x `auto`). Profiles also set `cv_folds`, `use_char` and `max_word_features` / `max_char_features` in `config.yaml` — see [Profiles](#profiles-configyaml).

**Resources (env):**
- **`APIV3_N_JOBS`** — CPU cores for head training (`-1` = all, joblib semantics), additionally bounded by:
- **`APIV3_CPU_MAX_PERCENT`** — hard CPU budget for a training run (default **60**): the effective thread count is `min(N_JOBS, cores × percent/100)`, so training never occupies more than ~60 % of the machine (measured: 55 % peak on 16 cores) and the API stays responsive. `100` disables the cap; `GET /config` shows the resolved `effective_n_jobs`.
- **`APIV3_SOLVER` / `APIV3_PARALLEL_BACKEND`** — default `newton-cg` + `threading`: converges fast, **keeps float32** and releases the GIL → all cores share **one** matrix.
- **`APIV3_TFIDF_MAX_WORD_FEATURES` / `_MAX_CHAR_FEATURES`** — global feature caps (defaults, overridable per profile).
- **`APIV3_WARMUP_MODELS` / `APIV3_MAX_MODELS_IN_MEMORY`** — serving speed. Name the models a server answers with (comma-separated) and they are loaded at startup and **stay resident**: the cache is sized to fit the list, so listing four models keeps four warm rather than the last two. Without it the first `/predict` after a model switch pays the cold load. `GET /config` reports the resolved `effective_max_models_in_memory`. Budget the RAM: a bundle's head is `n_labels × n_features × 4 bytes` (~48 MB for a 60-label model at 200 000 features). Both are wired through `docker-compose.yml` and the Helm chart (`config.limits.warmupModels`).
- **Responsiveness:** BLAS threads are capped to 1 in code (before numpy is imported) — this prevents thread oversubscription from nested parallelism (otherwise a frozen desktop UI / blocked API during training). Combined with the default 60 % CPU budget (`APIV3_CPU_MAX_PERCENT`), a training run leaves real headroom for the API and desktop out of the box.

See [`.env.example`](.env.example) for a ready-to-copy sample and [`docs/configuration.md`](docs/configuration.md) for the **complete reference** — every `APIV3_*` variable with its default, the `config.yaml` training options, per-request overrides, and quick setups for local (no auth) and production.

**Memory:** features are sparse + float32 throughout; intermediate matrices are released (`del`), and the matrix size (`dims`, `nnz`, MB) is logged during training. The solver matters: the default **`newton-cg` keeps float32**, so under the `threading` backend all cores share **one** matrix (no per-core copy, ~1× RAM). `lbfgs`/`liblinear` upcast to float64 per fit (≈2× the matrix).

**On 8 GB / large data:** thanks to the float32 solver + shared matrix, even large datasets run on **all cores**. Measured (15k rows, 200k dims): `newton-cg`/`threading` `n_jobs=8` ≈ **13 s** at ~1 GB fit RAM and F1 micro 0.81 — `lbfgs` (float64) needs ~4 GB for the same fit. For several 100k rows, the `fast` profile (word-only) shrinks the matrix further — see [sizing](#how-large-a-dataset-does-this-handle) for what that costs in robustness.

**Example run (30k rows, `auto` profile, all cores):** ~5 min end-to-end (load → 6× C search → deploy fit → save), **F1 micro ≈ 0.80 / macro ≈ 0.62** over 47 subjects, at **~0.5 GB RAM** — confirming full core utilization at low memory.

**Auto-optimization per run:** `C` (grid on validation) and the thresholds (global + per-label) are determined automatically. `min_samples_per_label` is **not** auto-scaled but declared — it defaults to `20` and is settable per request, because dropping labels is a decision worth seeing rather than a hidden heuristic. Send `null` for the old size-scaled behaviour (2 / 5 / 20 / 35). If no label reaches the value, training stops with a `status=error` on `/train/status` naming what the most frequent label actually has. Selection and reported metrics always measure the decision rule serving actually applies (`decision_rule` in the metrics): tuned thresholds for multilabel, argmax for multiclass/binary (threshold tuning is skipped there — serving never reads thresholds for single-label tasks).

### How large a dataset does this handle?

🟢 **Anchored on a real full-scale run** (`faecher_300k_auto`, 2026-07-26): 156 373 rows
× 60 labels from `data_300k.csv`, `auto` profile, 9 of 16 cores under the default 60 %
CPU budget.

| Measured at 156 373 rows (`auto`) | |
|---|---:|
| Total wall-clock, load → save | **40.2 min** |
| Non-zeros per document | 365 |
| Feature matrix (200 000 dims) | 436 MB |
| One head fit over all rows | ~4 min (C=2) … 5.8 min (C=32) |
| One vectorization pass over all rows | 2.3 min |
| Bundle save | **3 s** |

Row scaling is linear (`scripts/benchmark_row_scaling.py` measured exponent **1.00** for
memory and vectorization), so at constant label count 600 000 rows is 3.84× the above:
**~1.7 GB matrix** and **~2.6 h** for `auto`.

> ⚠️ An earlier revision of this section put `auto` at 600 k at ~1.2 h. That figure was
> derived twice over — an older `(3,5)` benchmark rescaled by a separately measured
> `(3,5) → (5,5)` speedup — and the run above shows it was **~2× too optimistic**: the
> per-fit speedup measured on 30 k rows does not carry to 600 k, where the head's
> `n_labels × 200 000` coefficients cost as much as the shrunken `nnz` saves. The
> numbers here replace it and are extrapolated in **one** step from a real run.

Non-zeros per document is what turns a large corpus into gigabytes — and it is
dataset-dependent, not a constant: 365 here against 278 measured on `data_30k_ai.csv`,
because this corpus has longer descriptions. Word-only (`fast`'s shape) drops it to ~70,
which is a ~5× smaller matrix for −0.0072 macro F1 clean but **3.6× worse degradation
under typos**.

**What that costs per profile**, scaled from the measured `auto` run above:

| Profile | Head fits¹ | Vectorization passes | 156 k rows | 600 k rows | Peak OOF buffer² |
|---------|-----------:|---------------------:|-----------:|-----------:|-----------------:|
| `fast` | 2.4 | 2 | ~4 min | ~15 min | — (holdout) |
| `auto` | 7 | 4 | **40 min** 🟢 | **~2.6 h** | 346 MB |
| `best` | 13 | 6 | ~1.2 h | ~4.6 h | 346 MB |

¹ `(folds − 1) × |C_grid| + 1`, in units of the full dataset — exact, it follows
directly from the loop in `tuning.cross_val_evaluate`. The 40 min for `auto` at 156 k is
🟢 measured; everything else scales from it, so the **ratio** is solid (`best` = 1.9×
`auto` in fits) while the absolute hours are estimates, not a schedule.
² k-fold CV holds one `rows × labels` float32 buffer **per `C` candidate** —
`600 000 × 48 × 4 bytes` = 115 MB each. Note this makes the *grid size*, not the fold
count, the memory lever under CV: folds are processed one at a time, candidates are not.

**Recommendation by size:** `auto` up to a few hundred thousand rows — it delivered the
run above in 40 min while still training the shipped model on 100 % of the data. At
600 k it is a half-day job rather than a coffee break, so plan it as one. `best` is
worth its ~2.4× on a model you will actually deploy, for two reasons: it closes the fold
gap, and it is the only rung that can tell whether the `C` optimum is genuinely
bracketed — the run above selected `C=32`, the **edge** of `auto`'s grid, which by this
project's own rule means the search ran out of candidates. Reach for `fast` only to
check that a pipeline runs at all.

> **`fast` is not the big-data answer.** It is word-only, which is measured to be a bad
> trade on real editorial text: it saves ~75 % of the matrix but degrades **3.6× more
> under character noise** (−0.0920 vs −0.0259 micro at 8 % typos). And because it
> evaluates on a holdout split, the model it ships never learns from the test share.
> If `auto` is too slow for your box, lower `cv_folds` to `2` before dropping to `fast`.

RAM planning beyond the matrix: the head is `n_labels × n_features × 4 bytes`
(independent of the row count) and the label matrix is `rows × labels` int8.

### Are the vocabulary caps cutting off signal?

The caps **are** binding: on `data_30k_ai.csv` the natural vocabulary is 134 835 word +
255 268 character n-grams, so the shipped 80 000 / 120 000 keeps 59 % and 47 % of it.
🟢 Measured whether the discarded tail matters (`scripts/benchmark_feature_caps.py`,
identical rows/split/C grid — only the caps change):

| Caps | Vocabulary built | Saturated | Head size | F1 macro | Δ macro | F1 micro | Fit time |
|------|-----------------:|:---------:|----------:|---------:|--------:|---------:|---------:|
| 40 k / 60 k | 40 000 / 60 000 | yes | **18.3 MB** | 0.7008 | −0.0052 | 0.7850 | 28.5 s |
| **80 k / 120 k (shipped)** | 80 000 / 120 000 | yes | 36.6 MB | **0.7060** | ±0 | 0.7910 | 37.8 s |
| 160 k / 240 k | 134 835 / 240 000 | no | 68.6 MB | 0.7055 | −0.0005 | 0.7911 | 63.7 s |
| 320 k / 480 k | 134 835 / 255 268 | no | 71.4 MB | 0.7051 | −0.0009 | 0.7904 | 60.3 s |

**Raising them does nothing.** Doubling exhausts the word vocabulary completely and
still lands at −0.0005 — noise. `max_features` keeps the most frequent terms, so what
gets cut is the rare tail, and with `min_df=2` already dropping hapaxes that tail
carries no signal. It costs, though: the head grows linearly with the feature count
(36.6 → 68.6 MB) and the fit time nearly doubles.

**Halving is the interesting direction.** 40 k / 60 k costs only −0.0052 macro and
**halves the head**. On a memory-constrained host that is a real trade to consider —
especially for a wide vocab, where the head dominates: a 300-label model is a 229 MB
head at 200 k features and ~115 MB at 100 k.

### Why TF-IDF (and not embeddings)?

On this keyword-heavy metadata, **TF-IDF + LogReg (F1 micro ≈ 0.80)** clearly beat static embeddings (Model2Vec ~0.60–0.73) and real transformer embeddings (e5-small + MLP ~0.78) — at a fraction of the cost (seconds instead of minutes, no torch, ~0.7 GB). An ensemble (TF-IDF ⊕ e5+MLP) was only ~1.4 points higher and does not justify the embedding stack.

🟢 **Re-checked as a feature union** (2026-07-25, `data_30k_ai.csv`, same head/split/threshold procedure — static Model2Vec vectors L2-normalised and concatenated onto the TF-IDF block, one LogReg head, not an ensemble):

| Features | F1 macro | Δ macro | F1 micro | nnz | Fit time |
|----------|---------:|--------:|---------:|----:|---------:|
| TF-IDF (deployed) | 0.7060 | ±0 | 0.7910 | 13.4 M | 35.4 s |
| static embedding only | 0.5518 | **−0.1542** | 0.6391 | 19.0 M | 29.5 s |
| TF-IDF ⊕ embedding (α=1.0) | 0.6984 | −0.0076 | 0.7817 | 32.4 M | 90.5 s |
| TF-IDF ⊕ embedding (α=0.3) | 0.7046 | −0.0014 | 0.7876 | 32.4 M | 77.1 s |

The union **never helps**: as the dense block's weight α shrinks, the result converges
back to the plain TF-IDF baseline — the head is effectively learning to ignore it. The
rationale for trying was sound (a dense semantic space is genuinely different from the
sparse lexical one, unlike a second linear head on the same features), but
complementarity requires the weaker source to be right where the stronger one is wrong.
On *this* data it is not: the text is title + keywords, which is TF-IDF's home turf and
exactly what a single averaged 1024-d vector blurs away. Cost if adopted anyway: 2.2×
fit time (the dense block triples the non-zeros) plus a runtime dependency that would
end the torch-free guarantee. *Caveat: one static model was tested (an e5-large
distillation); a stronger distillation would have to close a 15-point gap **and** then
add complementary signal.*

## Model interop / export

An exported model is a ZIP with `config.json` + `head.skops` + `vectorizer.skops` +
`vocabulary.json` (pure sklearn objects) and loads **directly with scikit-learn + skops** —
without this app. Two further members are generated per export and describe *that
archive*: a **`README.md`** model card (what it classifies, the author's own statements,
how it was trained, how well it scores, and its ten weakest labels) and a
**`manifest.json`** listing a SHA-256 for every other member. Import verifies the
manifest and refuses an archive that arrived altered or incomplete — an archive without
one still imports, so bundles shared before 3.2 keep working. Neither file is kept in the
installed bundle; both are rebuilt on the next export. The TF-IDF vocabularies live in `vocabulary.json` (terms in column
order) rather than inside the skops container: skops is super-quadratic in the number of
dict entries, which made a 200 000-term vocabulary take ~45 min and 148 MB to write
instead of 0.3 s and 3 MB. Re-attaching them is two lines:

```python
import json
import skops.io as sio
from scipy.sparse import hstack

cfg = json.load(open("config.json", encoding="utf-8"))
word_vec, char_vec = sio.load("vectorizer.skops", trusted=sio.get_untrusted_types(file="vectorizer.skops"))
head = sio.load("head.skops", trusted=sio.get_untrusted_types(file="head.skops"))

# The vocabularies ship separately — attach them before transforming.
for sub, terms in zip([word_vec, char_vec], json.load(open("vocabulary.json", encoding="utf-8"))):
    if sub is not None:
        sub.vocabulary_ = {term: index for index, term in enumerate(terms)}

texts = ["Bruchrechnung und Gleichungen lösen"]
X = word_vec.transform(texts)
if char_vec is not None:                 # word-only profiles have no char vectorizer
    X = hstack([X, char_vec.transform(texts)])
proba = head.predict_proba(X)            # column order == cfg["classes"]
# multilabel: apply cfg["per_label_thresholds"] (fallback cfg["global_threshold"]);
# multiclass/binary (cfg["task_type"]): take proba.argmax(axis=1) — thresholds are unused.
```

For hosting in other ML serving systems:
- **MLflow / BentoML / Ray Serve** can wrap the sklearn pipeline losslessly (recommended).
- **ONNX** (via `skl2onnx`) is possible, but TF-IDF with `char_wb` n-grams converts only partially.

## Deployment

- **Docker (local):** `docker compose up -d` — set `APIV3_API_KEY_ADMIN` / `APIV3_API_KEY_READONLY` in `.env` first; datasets and models persist in the `classification-data` volume.
- **Kubernetes:** Helm chart in [`deploy/helm/classification-api`](deploy/helm/classification-api/README.md) — StatefulSet with exactly **1 replica** (process-local state + one PVC), probes on `/health`, API keys via chart secret.
- **CI/CD:** GitHub Actions ([`.github/workflows/`](.github/workflows)) run the three quality gates and publish the image to GHCR; GitLab ([`.gitlab-ci.yml`](.gitlab-ci.yml)) mirrors that for self-hosted registries + Helm-chart push (credentials via CI/CD variables only).

## Tests

```bash
pip install -r requirements.txt -c requirements.lock
pip install -r requirements-dev.txt                # pinned pytest/httpx/ruff/mypy
python -m pytest tests -q                          # 198 tests
python -m ruff check app tests                     # lint
python -m mypy app --config-file pyproject.toml    # types
```

## Security & operations

- Models are pickle-free (skops); import rejects any file with unknown types, unexpected member names (allowlist of the four bundle files), and archives that inflate both past 64 MB and far beyond their upload size (zip-bomb guard). Bundles are written atomically (staged in a hidden tmp dir, then renamed), so a crash can never leave a half-readable model.
- Single-worker design (training status, model cache and rate limiter are process-local). Plan a shared store before running multiple workers.
- Rate limiting keys on the client IP and covers every expensive or public route: `/predict*`, `/train`, all import/export endpoints, the CSV-reading `GET /datasets`, `GET /datasets/{name}`, `/datasets/analyze` + `/datasets/{name}/validate`, and the key-less `GET /share/{id}` (throttles share-id brute-forcing). Cheap status routes and `/health` stay unthrottled by design (probes, UI polling). Behind a reverse proxy all clients share the proxy's IP, so limits act globally — run uvicorn with `--proxy-headers --forwarded-allow-ips <proxy>` (or swap the limiter key function) so real client IPs are used.
- Every response carries baseline security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`). No HSTS in-app — set it at the TLS-terminating reverse proxy.
- The Docker base image is digest-pinned; the Helm chart runs with `readOnlyRootFilesystem: true`, `runAsNonRoot`, dropped capabilities and `seccompProfile: RuntimeDefault`. Writable paths are the `/data` PVC (datasets + models) plus an `emptyDir` at `/tmp` (multipart uploads over ~1 MB spool there — it must be writable or uploads fail).
- Dependencies: `requirements.lock` pins the direct dependencies (the exact versions the test suite ran against); `requirements-hashes.lock` pins the **full transitive tree with sha256 hashes** (compiled from the lock via `uv pip compile --generate-hashes --universal`, targeting the image's Python 3.11). Docker and CI install with `--require-hashes --only-binary=:all:` — nothing unpinned or tampered with can enter the image. Update deliberately: bump the lock, re-run ruff/mypy/pytest, recompile the hashes file (command in the lock header).
- Docker: see `Dockerfile` (runs as non-root).
- **What to alert on** (Prometheus / uptime checks): `GET /health` non-200 (liveness); `apiv3_training_running == 1` for longer than your largest expected training run (stuck job); an unexpected drop of `apiv3_models_total` (lost volume/PVC); volume usage of the data mount (datasets + bundles grow). Error tracking: unhandled exceptions are logged server-side with full tracebacks (`api_v3.*` loggers) — ship container logs to your aggregator.

## License

[MIT](LICENSE)
