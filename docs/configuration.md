# Configuration reference

Everything the app reads at runtime, in one place. Three layers:

1. **Environment variables** (prefix `APIV3_`, optionally from a `.env` file) — runtime
   behaviour: paths, auth, limits, compute. Copy [`.env.example`](../.env.example)
   to `.env` and adjust; every variable below has a safe default, so an empty
   environment also works for local use.
2. **`config.yaml`** — *training* behaviour: quality profiles, preprocessing, split.
   Reloaded on every `POST /train` (no restart needed).
3. **Per-request fields** — a few settings can be overridden per training request
   (`cv_folds`, `min_samples_per_label`, `task_type`); the request wins over the file.

## Quick setups

**Local, no auth** (Swagger `/docs` and the admin UI `/ui` work without signing in):

```bash
APIV3_AUTH_ENABLED=false uvicorn app.main:app --port 8000
```

**Production minimum:**

```bash
cp .env.example .env
# set strong values for the two keys:
#   APIV3_API_KEY_ADMIN=...      APIV3_API_KEY_READONLY=...
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Checklist for exposed deployments: strong keys set · TLS terminated at a reverse
proxy (set HSTS there) · `--proxy-headers --forwarded-allow-ips <proxy>` so rate
limits see real client IPs · storage paths on a persistent volume.

## Environment variables

### Storage

| Variable | Default | Description |
|---|---|---|
| `APIV3_DATA_DIR` | `./data` | CSV datasets (uploads land here). |
| `APIV3_MODELS_DIR` | `./models` | Trained model bundles. |
| `APIV3_SHARE_LINKS_FILE` | `./share_links.json` | Persisted expiring share links. |
| `APIV3_CONFIG_FILE` | `./config.yaml` | Training profiles file (layer 2 above). |

In containers, point all four at the mounted volume — the provided
`docker-compose.yml` and Helm chart do this automatically (`/data/*`).

### Authentication & UI

| Variable | Default | Description |
|---|---|---|
| `APIV3_AUTH_ENABLED` | `true` | `false` = full access without a key (local use only). |
| `APIV3_API_KEY_ADMIN` | – | Admin key: `/train`, `/train/stop`, dataset/model import, export, delete, analyze. |
| `APIV3_API_KEY_READONLY` | – | Readonly key: `/predict*`, `/train/status`, listings. |
| `APIV3_UI_ENABLED` | `true` | Serve the admin UI at `/ui`. The page itself is public like `/docs`; every data request carries the key the user signs in with (with auth disabled, the UI skips the sign-in). |

No key is ever required for `/health`, `/metrics` and `GET /share/{id}`
(bearer-capability links).

### Requests & limits

| Variable | Default | Description |
|---|---|---|
| `APIV3_CORS_ALLOW_ORIGINS` | *(empty)* | Comma-separated browser-origin allowlist; empty = no cross-origin access. |
| `APIV3_MAX_UPLOAD_MB` | `200` | Upload cap for datasets/model bundles (streamed; aborts at the cap). |
| `APIV3_MAX_MODELS_IN_MEMORY` | `2` | LRU cache size: models kept resident in RAM. |
| `APIV3_WARMUP_MODELS` | *(empty)* | Comma-separated model names to preload + warm on startup so their first `/predict` pays no cold skops-load. Best-effort (a missing name is logged and skipped); keep the count ≤ `MAX_MODELS_IN_MEMORY`. |
| `APIV3_RATE_LIMIT_ENABLED` | `true` | In-process limiter, keyed on client IP. |
| `APIV3_RATE_LIMIT_PREDICT` | `300/minute` | `/predict`, `/predict/batch`, `/predict/multi`, `/predict/explain`. |
| `APIV3_RATE_LIMIT_TRAIN` | `5/minute` | `POST /train`. |
| `APIV3_RATE_LIMIT_EXPORT` | `30/minute` | Import/export endpoints, CSV-reading dataset endpoints, public `GET /share/{id}`. |
| `APIV3_RATE_LIMIT_DEFAULT` | `120/minute` | Fallback bucket. |

### Compute (training resources)

| Variable | Default | Description |
|---|---|---|
| `APIV3_N_JOBS` | `-1` | Threads for the label-wise head fits (joblib semantics: `-1` = all cores, `-2` = all but one). Bounded by the CPU budget below. |
| `APIV3_CPU_MAX_PERCENT` | `60` | Hard CPU budget for a training run: effective threads = `min(N_JOBS, cores × percent/100)`. Keeps the API responsive; `100` disables the cap. `GET /config` shows the resolved value. |
| `APIV3_SOLVER` | `newton-cg` | LogisticRegression solver. `newton-cg`/`saga` keep float32 and release the GIL (all cores share ONE matrix); `lbfgs`/`liblinear` upcast to float64. |
| `APIV3_PARALLEL_BACKEND` | `threading` | joblib backend for the per-label fits. |
| `APIV3_TFIDF_MAX_WORD_FEATURES` | `80000` | Word-n-gram vocabulary cap (main RAM lever; profiles may override). |
| `APIV3_TFIDF_MAX_CHAR_FEATURES` | `120000` | Char-n-gram vocabulary cap. |
| `APIV3_RANDOM_SEED` | `42` | Split/CV seed (reproducibility). |

BLAS threads are pinned to 1 in code before numpy loads (prevents thread
oversubscription); override only via real OS environment variables
(`OMP_NUM_THREADS` etc.).

### Logging

| Variable | Default | Description |
|---|---|---|
| `APIV3_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. Failed trainings log full tracebacks server-side; clients get sanitized messages. |

## `config.yaml` (training profiles)

```yaml
default_profile: auto
profiles:
  auto:       # C_grid, tune_threshold, threshold_per_label, use_char,
  fast:       # max_word_features, max_char_features per profile
  thorough:
preprocessing:
  min_text_length_chars: 5
  drop_duplicates: true          # identical texts are deduplicated (prevents CV leakage)
  min_samples_per_label: null    # null = auto-scaled to dataset size
split:
  validation_size: 0.15
  test_size: 0.15
  cv_folds: 0                    # 0 = train/val/test split; >=2 = k-fold CV
```

- **Profiles** are freely extensible; `optimize_parameters` in the train request
  selects one by name.
- **`cv_folds`** can be overridden per request (`cv_folds` body field: `0` = split,
  `2–20` = k-fold CV — every row trains AND validates via out-of-fold, the deployed
  model is fit on 100 % of the data).
- **`min_samples_per_label`** can also be set per request.

## Where to see the effective configuration

`GET /config` (readonly key) returns the non-sensitive effective values, including
the resolved `effective_n_jobs` a training run will actually use.
