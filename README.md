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

1. Put a CSV into `data/` (or upload via `POST /datasets/import`).
2. `POST /train` with `text_columns`, `label_column`, a profile (`auto`/`fast`).
3. `GET /train/status` until `completed` (includes test metrics).
4. `POST /predict` with `model_name`.
5. Multiple tasks = multiple models: select per request via `model_name`.

### Quick example (curl)

```bash
# Train (admin key) -> poll status -> predict (any valid key).
curl -X POST localhost:8000/train -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" -d '{
  "dataset_name": "data_30k.csv", "model_name": "subjects",
  "text_columns": ["properties.cclom:title","properties.cclom:general_description","properties.cclom:general_keyword"],
  "label_column": "properties.ccm:taxonid", "optimize_parameters": "auto",
  "label_filter": "http://w3id.org/openeduhub/vocabs/discipline/"
}'
curl localhost:8000/train/status -H "X-API-Key: $RO_KEY"
curl -X POST localhost:8000/predict -H "X-API-Key: $RO_KEY" -H "Content-Type: application/json" \
  -d '{"texts": ["Bruchrechnung und Gleichungen lösen"], "model_name": "subjects"}'
```

With `APIV3_AUTH_ENABLED=false` the `X-API-Key` header is optional.

## Profiles (`config.yaml`)

| Profile | Description |
|---------|-------------|
| `auto` | Automatic `C` selection (grid up to 16 on validation) + per-label threshold tuning *(recommended)* |
| `fast` | Word n-grams only, few `C` values, global threshold — fastest run, lowest RAM |
| `thorough` | Widest `C` grid + per-label thresholds — best quality, slowest run |

Add your own profiles in `config.yaml` (fields: `C_grid`, `tune_threshold`, `threshold_per_label`, `use_char`, `max_word_features`, `max_char_features`).

## Endpoints (excerpt)

- **Training:** `POST /train`, `GET /train/status`, `POST /train/stop`, `GET /train/profiles`
- **Classification:** `POST /predict`, `POST /predict/batch`, `POST /predict/multi` (several models = target fields in one call; each model applies its own tuned thresholds, evaluation stays per model), `POST /predict/explain`. All predict endpoints can attach `baseline_diff` (`include_baseline_diff=true`; always on in `/predict/explain`): confidence minus the model's empty-text prediction — a high confidence with a diff near zero means the label fires for almost anything, not for this text.
- **Models:** `GET /models`, `GET /models/{name}`, `DELETE /models/{name}`, `POST /models/{name}/export`, `POST /models/import`, `GET /share/{id}`
- **Datasets:** `GET /datasets`, `GET /datasets/{name}`, `POST /datasets/analyze`, `POST /datasets/{name}/validate`, import/export/delete
- **System:** `GET /health`, `GET /config`, `GET /metrics` (Prometheus text format, public — operational gauges only: uptime, model counts, training status)

## Status & monitoring

`GET /train/status` returns live: `status` (idle/running/completed/error/stopped), `phase` (loading → preparing → features → selecting → threshold → evaluating → saving → done), `phase_detail` (e.g. `Testing regularization: C=8 (5/6) – best F1 so far 0.629`, or `Fold 3/5: C=2.0 (14/30 fits)` in CV mode), `progress` (0–100), `message`, `elapsed_seconds`, `eta_seconds` (estimated time remaining), `model_name`, `results` (metrics on completion), `error`. **Progress moves within the long phases** — the C search spreads over 55→75 % and a cross-validation run over 45→90 % (one step per head fit), so the ETA stays meaningful instead of freezing at a phase boundary. Cancel via `POST /train/stop` (`hard=false`/`true`); it takes effect promptly (checked between the C fits and before deployment). The admin UI shows the same status as a live progress card on the Training tab plus a compact chip in the top bar that stays visible on every tab while a run is active.

For Prometheus, `GET /metrics` exposes operational gauges (uptime, models on disk / in the LRU cache, training running + progress) in the text exposition format — hand-rolled, no client-library dependency. The Helm chart's ServiceMonitor (`global.metrics.servicemonitor.enabled`) scrapes it.

## Resources & tuning

**Fast ↔ good** via the profile (`optimize_parameters`): `fast` (word-only TF-IDF, smaller vocabulary cap → low RAM, fast), `auto` (word + char, recommended), `thorough` (wider C grid, best quality). Profiles also set `use_char` and `max_word_features` / `max_char_features` in `config.yaml`.

**Resources (env):**
- **`APIV3_N_JOBS`** — CPU cores for head training (`-1` = all, joblib semantics), additionally bounded by:
- **`APIV3_CPU_MAX_PERCENT`** — hard CPU budget for a training run (default **60**): the effective thread count is `min(N_JOBS, cores × percent/100)`, so training never occupies more than ~60 % of the machine (measured: 55 % peak on 16 cores) and the API stays responsive. `100` disables the cap; `GET /config` shows the resolved `effective_n_jobs`.
- **`APIV3_SOLVER` / `APIV3_PARALLEL_BACKEND`** — default `newton-cg` + `threading`: converges fast, **keeps float32** and releases the GIL → all cores share **one** matrix.
- **`APIV3_TFIDF_MAX_WORD_FEATURES` / `_MAX_CHAR_FEATURES`** — global feature caps (defaults, overridable per profile).
- **Responsiveness:** BLAS threads are capped to 1 in code (before numpy is imported) — this prevents thread oversubscription from nested parallelism (otherwise a frozen desktop UI / blocked API during training). Combined with the default 60 % CPU budget (`APIV3_CPU_MAX_PERCENT`), a training run leaves real headroom for the API and desktop out of the box.

See [`.env.example`](.env.example) for a ready-to-copy sample and [`docs/configuration.md`](docs/configuration.md) for the **complete reference** — every `APIV3_*` variable with its default, the `config.yaml` training options, per-request overrides, and quick setups for local (no auth) and production.

**Memory:** features are sparse + float32 throughout; intermediate matrices are released (`del`), and the matrix size (`dims`, `nnz`, MB) is logged during training. The solver matters: the default **`newton-cg` keeps float32**, so under the `threading` backend all cores share **one** matrix (no per-core copy, ~1× RAM). `lbfgs`/`liblinear` upcast to float64 per fit (≈2× the matrix).

**On 8 GB / large data:** thanks to the float32 solver + shared matrix, even large datasets run on **all cores**. Measured (15k rows, 200k dims): `newton-cg`/`threading` `n_jobs=8` ≈ **13 s** at ~1 GB fit RAM and F1 micro 0.81 — `lbfgs` (float64) needs ~4 GB for the same fit. For several 100k rows, the `fast` profile (word-only) shrinks the matrix further (~−60 % dimensions).

**Example run (30k rows, `auto` profile, all cores):** ~5 min end-to-end (load → 6× C search → deploy fit → save), **F1 micro ≈ 0.80 / macro ≈ 0.62** over 47 subjects, at **~0.5 GB RAM** — confirming full core utilization at low memory.

**Auto-optimization per run:** `C` (grid on validation), thresholds (global + per-label) and `min_samples_per_label` are determined automatically (the last is also settable per request).

### Why TF-IDF (and not embeddings)?

On this keyword-heavy metadata, **TF-IDF + LogReg (F1 micro ≈ 0.80)** clearly beat static embeddings (Model2Vec ~0.60–0.73) and real transformer embeddings (e5-small + MLP ~0.78) — at a fraction of the cost (seconds instead of minutes, no torch, ~0.7 GB). An ensemble (TF-IDF ⊕ e5+MLP) was only ~1.4 points higher and does not justify the embedding stack.

## Model interop / export

An exported model is a ZIP with `config.json` + `head.skops` + `vectorizer.skops` (pure sklearn objects) and loads **directly with scikit-learn + skops** — without this app:

```python
import json
import skops.io as sio
from scipy.sparse import hstack

cfg = json.load(open("config.json", encoding="utf-8"))
word_vec, char_vec = sio.load("vectorizer.skops", trusted=sio.get_untrusted_types(file="vectorizer.skops"))
head = sio.load("head.skops", trusted=sio.get_untrusted_types(file="head.skops"))

texts = ["Bruchrechnung und Gleichungen lösen"]
X = word_vec.transform(texts)
if char_vec is not None:                 # word-only profiles have no char vectorizer
    X = hstack([X, char_vec.transform(texts)])
proba = head.predict_proba(X)            # column order == cfg["classes"]
# Apply cfg["per_label_thresholds"] (fallback cfg["global_threshold"]) to turn proba into labels.
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
python -m pytest tests -q                          # 93 tests (~94 % line coverage)
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
