# Changelog

Notable changes to MetaClassify (torch-free metadata text-classification API). Dates are UTC.

## [Unreleased] — audit remediation (2026-07-16)

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
  and the previously-dead `rate_limit_default` is wired; `separator` is capped to
  one character (ReDoS); the 429 body now uses the shared `{"detail": ...}` envelope.

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

## [Unreleased] — hardening & evaluation wave (2026-07-07 … 2026-07-08)

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
