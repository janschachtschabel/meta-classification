# Lightweight, CPU-only image for MetaClassify (no GPU, no torch).
# Digest-pinned (supply chain): the tag alone is mutable. Dependabot proposes a new digest
# when the tag moves (.github/dependabot.yml); to look by hand:
#   docker buildx imagetools inspect python:3.11-slim-trixie   (take the index digest)
FROM python:3.11-slim-trixie@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

WORKDIR /app

# No compiler needed: every runtime dep ships prebuilt manylinux wheels.
# requirements-hashes.lock pins the FULL transitive tree with sha256 hashes
# (compiled from requirements.txt, constrained to the tested requirements.lock
# versions) — pip refuses anything unpinned or tampered with. The base image's own pip
# installs it: upgrading pip first fetched whatever pip was newest, unpinned and unhashed,
# into every build (audit 2026-09-30, B10).
COPY requirements-hashes.lock /app/
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements-hashes.lock

COPY app/ /app/app/
COPY config.yaml /app/
# The two bundle repairs run where the bundles are -- here, via `docker exec` or
# `kubectl exec` (audit 2026-09-30, B06). They find the volume through the ENV below.
COPY scripts/patch_bundle_labels.py scripts/prune_bundle_labels.py /app/scripts/

# The code stays root's: the process that serves requests owns its data and nothing else.
# `chown -R appuser /app` let it rewrite its own code (audit 2026-09-30, B10).
RUN useradd -m -u 1000 appuser \
    && mkdir -p /data/datasets /data/models \
    && chown -R appuser:appuser /data
USER appuser

# Every writable path on the volume, as compose and the chart set them: the defaults are
# files beside the code, which an unconfigured `docker run` wrote into the container's own
# layer -- lost with the container, and no longer writable at all now that /app is root's.
ENV APIV3_DATA_DIR=/data/datasets \
    APIV3_MODELS_DIR=/data/models \
    APIV3_SHARE_LINKS_FILE=/data/share_links.json \
    APIV3_FEEDBACK_FILE=/data/feedback.jsonl \
    APIV3_JOB_HISTORY_FILE=/data/job_history.jsonl

# Threads/resources: 1 BLAS thread per worker (joblib parallelises across labels,
# so nested BLAS threads would only oversubscribe the CPU). APIV3_N_JOBS=auto takes
# every core the container grants, bounded by the default APIV3_CPU_MAX_PERCENT=60
# budget (set 100 to disable) so the API stays responsive while a training job runs,
# and per head fit by the memory budget (APIV3_TRAIN_MEMORY_MB, default auto).
# All overridable at `docker run -e VAR=...`.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    APIV3_N_JOBS=auto

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --proxy-headers: trust X-Forwarded-For so the rate limiter keys on the real
# client IP, not the reverse proxy's. Set FORWARDED_ALLOW_IPS (env, uvicorn reads
# it) to the proxy's source address/range — the default 127.0.0.1 trusts nothing
# from a pod-network proxy.
# --workers 1: the training job, model cache, rate limits and share links live in ONE
# process, and without the flag uvicorn takes its worker count from WEB_CONCURRENCY --
# with several, share links made in one worker were unknown to the next (audit
# 2026-09-30, R07). Scale this app with CPU and memory, not processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--workers", "1"]
