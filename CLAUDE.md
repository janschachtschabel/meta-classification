# Project: MetaClassify (classification-api v3)

> **Naming:** the product is **MetaClassify** on GitHub / in all user-facing text
> (app title, README, docs, UI). The internal directory (`api_v3`), Python package
> (`app`), env prefix (`APIV3_`), Helm chart name and container image path keep
> their existing identifiers — renaming those would break deployments/config.

Lightweight, secure text-classification API for metadata (TF-IDF + OneVsRest
LogisticRegression, FastAPI, CPU-only, torch-free). This directory is the repo
root when published standalone.

## Tech stack
- Python >= 3.11 (Docker base + hashes lock target 3.11) · FastAPI + uvicorn · scikit-learn (sparse float32 TF-IDF word+char, solver `newton-cg`) · skops (pickle-free persistence) · pydantic-settings (env prefix `APIV3_`)

## Commands (run from the repo root)
- Test: `python -m pytest tests -q`
- Lint: `python -m ruff check app tests scripts`
- Types: `python -m mypy app --config-file pyproject.toml`
  (mypy does NOT auto-discover the config from other directories — always pass `--config-file`.)

## Conventions
- All comments, docstrings, and API texts are English; comments explain *why*, not *what*.
- Bug fixes are test-first: failing test → fix → green. Never weaken a test to pass.
- Security: no pickle (skops only); model/dataset import via file upload only (no URL fetch); user-supplied names go through `security.safe_name`; API keys compare in constant time.
- Dependencies: 13 runtime deps, pinned in `requirements.lock`; no new dependency without written justification.
- Files stay under ~300 lines; split by responsibility, not by line count.
- Single-worker design: training job, model LRU cache, and rate limiter are process-local. Do not introduce multi-worker assumptions (also: Helm chart is fixed at 1 replica).

## Architecture
- `app/routes/*` stay thin (auth, validation, HTTP mapping) and delegate to core modules (`data`, `dataset_stats`, `vectorizers`, `classifier`, `tuning`, `prepare`, `deploy`, `training`, `explain`, `registry`, `model_io`, `jobs`, `sharing`, `errors`).
- Training pipeline is split by stage: `prepare.py` (load/clean/targets/split) → `deploy.py` (C-selection, thresholds, metrics, deploy fit) → `training.py` (orchestration + metadata).
- Dataset code splits by responsibility: `data.py` = load/clean/targets/split core; `dataset_stats.py` = read-only inspection/statistics for the API (consumes `data`, never the reverse).
- `app/__init__.py` caps BLAS threads to 1 **before** numpy is imported — keep that the first thing the package does.
- Model bundles (format 2) = `config.json` + `metrics.json` + `head.skops` + `vectorizer.skops` + `vocabulary.json`. The TF-IDF vocabularies deliberately live OUTSIDE skops: it is super-quadratic in dict entries (200k terms = ~45 min / 148 MB vs 0.3 s / 3 MB as JSON). `model_io.py` owns the pickle-free (de)serialization + skops load-safety; `registry.py` owns the cache, disk locks and atomic publish (hidden `.name.tmp` dir + rename).

## Deployment
- Docker: `Dockerfile` (non-root, healthcheck) · local: `docker compose up -d` (keys via `.env`).
- Kubernetes: Helm chart in `deploy/helm/classification-api` (StatefulSet, 1 replica, `/data` PVC).
- CI/CD: GitHub Actions (`.github/workflows/` — gates + GHCR image) · GitLab (`.gitlab-ci.yml` — lint/test/build/helm, credentials via CI variables only).
