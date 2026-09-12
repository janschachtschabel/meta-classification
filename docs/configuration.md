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
| `APIV3_DATA_DIR` | `./data` | CSV datasets (uploads land here). `.csv` and `.csv.gz` are both accepted — gzip is read natively and is the sane choice above ~100 MB. |
| `APIV3_MODELS_DIR` | `./models` | Trained model bundles. |
| `APIV3_SHARE_LINKS_FILE` | `./share_links.json` | Persisted expiring share links. |
| `APIV3_FEEDBACK_FILE` | `./feedback.jsonl` | Corrections recorded via `POST /feedback` — not a log but the data a later run learns from, and never pruned. |
| `APIV3_JOB_HISTORY_FILE` | `./job_history.jsonl` | How finished runs ended (`GET /train/history`), newest 200. |
| `APIV3_CONFIG_FILE` | `./config.yaml` | Training profiles file (layer 2 above). |

Five of these are *state*: `DATA_DIR`, `MODELS_DIR`, `SHARE_LINKS_FILE`,
`FEEDBACK_FILE` and `JOB_HISTORY_FILE`. The provided `docker-compose.yml` and
Helm chart point the **first three** at the mounted volume (`/data/*`)
automatically; the two `.jsonl` files keep their default next to the code, which
in a container means inside the image. **Set them explicitly** if corrections and
run history are to survive — they are the two that are easiest to lose unnoticed:

```
APIV3_FEEDBACK_FILE=/data/feedback.jsonl
APIV3_JOB_HISTORY_FILE=/data/job_history.jsonl
```

Without that, a re-created container starts with an empty history and no
corrections, and under the Helm chart's read-only root filesystem
(`securityContext.readOnlyRootFilesystem: true`) the write fails outright: a
`POST /feedback` answers 500, while the history only logs a warning and stays
empty. The volume is the only thing that cannot be rebuilt from the image — see
[Backup & restore](../README.md#backup--restore) for what it costs to lose and
how to copy it consistently while the service runs. `APIV3_CONFIG_FILE` deliberately stays at the
image-baked `/app/config.yaml`: the profiles file ships with the image, and
under the Helm chart's read-only root filesystem it is **immutable at runtime**
(the "reloaded on every `POST /train`" note above then only matters for local
runs). To edit profiles in production, set `APIV3_CONFIG_FILE=/data/config.yaml`
and seed that file on the volume once.

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
| `APIV3_MAX_MODELS_IN_MEMORY` | `2` | LRU cache size: models kept resident in RAM. Raised automatically to fit `WARMUP_MODELS`; `GET /config` reports the resolved `effective_max_models_in_memory`. |
| `APIV3_WARMUP_MODELS` | *(empty)* | Comma-separated model names to preload + warm on startup so their first `/predict` pays no cold skops-load — the practical setting for a server answering several target fields. **All listed models stay resident** (the cache is sized to fit them). Best-effort: a missing or unloadable name is logged and skipped, never fatal to startup. Wired through `docker-compose.yml` and the Helm chart (`config.limits.warmupModels`). |
| `APIV3_RATE_LIMIT_ENABLED` | `true` | In-process limiter, keyed on client IP. |
| `APIV3_RATE_LIMIT_PREDICT` | `300/minute` | `/predict`, `/predict/batch`, `/predict/multi`, `/predict/explain`. |
| `APIV3_RATE_LIMIT_TRAIN` | `5/minute` | `POST /train`. |
| `APIV3_RATE_LIMIT_EXPORT` | `30/minute` | Import/export endpoints, CSV-reading dataset endpoints, public `GET /share/{id}`. |
| `APIV3_RATE_LIMIT_DEFAULT` | `120/minute` | Fallback bucket. |

### Compute (training resources)

| Variable | Default | Description |
|---|---|---|
| `APIV3_N_JOBS` | `auto` | Threads for the label-wise head fits. `auto` = every core this process may use; a number asks for that many (joblib semantics: `-1` = all cores, `-2` = all but one). "Cores" is container-aware: the cgroup CPU quota (a Kubernetes/Docker CPU limit) and the scheduler affinity mask bound `os.cpu_count()`, so `auto` inside a 4-CPU-limited pod means 4, not the node's core count. Bounded by the CPU budget below and, per fit, by the memory budget. |
| `APIV3_CPU_MAX_PERCENT` | `60` | Hard CPU budget for a training run: effective threads = `min(N_JOBS, available cores × percent/100)`. Keeps the API responsive; `100` disables the cap. `GET /config` shows the resolved value. |
| `APIV3_TRAIN_MEMORY_MB` | `auto` | Memory budget for a training run, in MiB. It bounds the head-fit threads: every concurrent per-label fit holds ~2.5× the feature matrix in solver buffers, so a run that would outgrow the budget trains with fewer threads — slower, same model — instead of being OOM-killed. `auto` = 85 % of the container's cgroup memory limit (Docker `--memory`, a Kubernetes memory limit) when there is one, otherwise no cap; `0` = no cap. `GET /config` shows the resolved `effective_train_memory_mb`. |
| `APIV3_TRAINING_ISOLATION` | `process` | Where a training runs. `process` = in a child process (`python -m app.train_worker`, JSON over pipes, nothing pickled): when the run ends, all of its memory goes back to the OS instead of staying with the process that serves requests. An OOM kill ends the run with an error that names the likely cause, instead of taking the API down: the child raises its own `oom_score_adj`, so the kernel picks it and not the API. (On Kubernetes that holds only where the kubelet does not kill the whole container — since 1.28 it does on cgroup v2, unless `singleProcessOOMKill` is set, 1.32+; there `APIV3_TRAIN_MEMORY_MB` is what keeps a run inside the limit.) A hard stop ends the run at once. `APIV3_TRAIN_MEMORY_MB` still covers both processes: the child counts the API process' memory as held. The cost is an interpreter start per run and a cold first `/predict` of the new model. `thread` = inside the API process, as before (what the test suite uses). `GET /config` shows it. |
| `APIV3_SOLVER` | `newton-cg` | LogisticRegression solver. `newton-cg`/`saga` keep float32 and release the GIL (all cores read ONE input matrix; each concurrent fit adds ~2–2.5× the matrix in solver buffers, which `APIV3_TRAIN_MEMORY_MB` accounts for); `lbfgs`/`liblinear` upcast to float64. |
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
  fast:       # C_grid, cv_folds, tune_threshold, threshold_per_label, use_char,
  auto:       # max_word_features, max_char_features per profile
  best:
preprocessing:
  min_text_length_chars: 5
  drop_duplicates: true          # identical texts are deduplicated (prevents CV leakage)
  min_samples_per_label: null    # null = auto-scaled to dataset size
  text_column_weights:           # default field weighting; {} in a request disables it
    properties.cclom:title: 2
    properties.cclom:general_keyword: 2
split:
  validation_size: 0.15
  test_size: 0.15
  cv_folds: 0                    # last-resort fallback only — see below
```

- **Profiles** are the *fast ↔ good* dial and ship as three rungs ordered by cost:

  | Profile | Char n-grams | `C_grid` | Evaluation | Head fits¹ | Deploys on |
  |---|---|---|---|---:|---|
  | `fast` | no | `[2, 32]` | holdout split | 2.4 | 85 % of rows |
  | `auto` | (5,5) | `[2, 8, 32]` | 3-fold CV | 7 | **100 %** |
  | `best` | (5,5) | `[2, 8, 32]` | 5-fold CV | 13 | **100 %** |

  ¹ in units of the full dataset: `(folds − 1) × |C_grid| + 1`, which is what drives
  wall-clock. `best` therefore costs ~1.9× `auto`.

  They are freely extensible; `optimize_parameters` in the train request selects one
  by name. **The `C` range is fixed at `2 … 32` for all of them**, from two measurements:

  - **Both ends are needed.** This project's targets optimize at *opposite* ends (subject
    picks `32`, educational level picked `2`), so `[4, 16]` would have two candidates and
    reach neither. Pinned by `test_every_profile_brackets_both_known_optima`.
  - **Above 32 quality drops.** Measured on a holdout split for both targets: macro F1
    peaks at `C=32` (school 0.7531, university 0.7814) and declines at 128 / 512 / 2048
    while fit time rises ~50 %. Pinned by
    `test_no_c_grid_reaches_past_the_measured_useful_range`.

  Within that range, resolution is cheap: a 3-candidate grid at 4× steps selects the same
  `C` at identical F1 as an 8-candidate grid at 2× steps. `auto` and `best` therefore share
  one grid, and the *only* difference between them is the fold count. Each bundle records
  the grid it searched as `c_grid` next to `best_C`; an edge pick there is a tie on a flat
  plateau, not a missing optimum (at fixed 5-fold CV, `C=32` and `C=128` scored 0.8135 vs
  0.8130).
- **`cv_folds`** resolves most-specific-first: **request > profile > `split.cv_folds`**.
  Every shipped profile sets its own, so the `split.cv_folds` above only applies to
  custom profiles that leave it unset. Per request: `0` = holdout split, `2–20` =
  k-fold CV. The distinction that matters is not accuracy but *data usage* — under
  k-fold CV every row trains AND validates (out-of-fold) and the deployed model is
  fit on 100 % of the data, while a holdout split permanently spends its test share
  on measurement. `k` only controls how much data the evaluation models see
  (`k=3` → 67 %, `k=5` → 80 %), so fewer folds bias the reported score slightly
  *pessimistic*, not optimistic.
- **`min_samples_per_label`** is set per request and **defaults to `20`** there
  (the `null` above only applies when a request omits nothing — the request field
  wins). Send `null` explicitly for the size-scaled heuristic (2 / 5 / 20 / 35).
  A value no label reaches aborts the run with `status=error` on `/train/status`,
  naming how many rows the most frequent label actually has.
- **`text_column_weights`** repeats a text column when the training text is assembled —
  `{"title": 2}` gives short, dense fields their term frequency back against a long
  description. The `preprocessing` default above ships as title + keywords at 2× (the
  measured optimum) and applies when a request omits the field, narrowed to the columns
  that request trains on; a request mapping overrides it and `{}` disables it.
  `GET /train/profiles` reports the active default. Training-time only: build the text
  you send to `/predict` the same way (see README).
- **`max_word_features` / `max_char_features`** are settable per request too, on top of
  the profile and the `APIV3_TFIDF_MAX_*_FEATURES` env vars (most specific wins). They
  are the main RAM lever, so they are bounded at 2 000 000. A model whose
  `tfidf.n_features` equals `max_word_features + max_char_features` had its vocabulary
  **truncated** — both values are recorded in the bundle metadata so that stays checkable.

## `data/label_names.json` — authoritative label display names (optional)

A plain `{"<label uri>": "<display name>"}` sidecar in the data directory. If present,
training uses it for the bundle's `uri_to_label` and it **overrides** names derived from
the CSV's `_DISPLAYNAME` column. Absent or malformed → ignored with a log warning; a
training run is never failed over a display name.

It exists because a CSV that separates label URIs *and* display names with the same
character is ambiguous whenever a name contains that character
(`"Rechts-, Wirtschafts- und Sozialwissenschaften"`). Names are then recovered from the
CSV only where the counts provably line up, plus a reconstruction pass — measured on
`data_300k.csv` that covers 70 % of labels, and the rest would otherwise show no name.

```bash
python scripts/fetch_vocab_labels.py                        # -> data/label_names.json
python scripts/patch_bundle_labels.py --apply               # repair EXISTING bundles
```

`fetch_vocab_labels.py` downloads SKOS vocabularies (add URLs at the top of the file) and
is **build-time only** — `app/` never fetches a URL, which is the same SSRF boundary that
makes dataset/model import upload-only. `patch_bundle_labels.py` rewrites `uri_to_label`
in a trained bundle's `config.json` **without retraining**: it is presentation-only data,
so the script asserts `classes`, thresholds and the skops members stay byte-identical and
keeps a `config.json.bak`. Run it without `--apply` first for a dry run.

## Container labels are never trained

A label value ending in `/` identifies a namespace, not a concept, and is dropped during
loading (`data.split_labels`). This is not configurable, because such a value is always a
tagging accident: measured, the bare vocabulary root `…/vocabs/discipline/` sat on 522 rows
and trained as a class scoring F1 0.4096 — predictable, meaningless, and beyond the reach of
`min_samples_per_label`. The guard is intentionally narrow: a genuine broader concept has an
id (`…/discipline/120`) and is kept, since the label hierarchy carries real signal.

Bundles trained before this guard still contain such a class. Loading one logs a warning
naming it; repair it in place with

```bash
python scripts/prune_bundle_labels.py --model <name> --apply    # omit --apply for a dry run
```

which drops the class *and* its estimator, verifies the remaining probabilities are
bit-identical, recomputes `f1_macro`, and keeps a full bundle backup.

## Where to see the effective configuration

`GET /config` (readonly key) returns the non-sensitive effective values, including
the resolved `effective_n_jobs` a training run will actually use.
