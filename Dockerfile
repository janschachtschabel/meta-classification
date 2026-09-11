# Lightweight, CPU-only image for MetaClassify (no GPU, no torch).
# Digest-pinned (supply chain): the tag alone is mutable. Update deliberately via
#   docker buildx imagetools inspect python:3.11-slim   (take the index digest)
FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

WORKDIR /app

# No compiler needed: every runtime dep ships prebuilt manylinux wheels.
# requirements-hashes.lock pins the FULL transitive tree with sha256 hashes
# (compiled from requirements.txt, constrained to the tested requirements.lock
# versions) — pip refuses anything unpinned or tampered with.
COPY requirements-hashes.lock /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements-hashes.lock

COPY app/ /app/app/
COPY config.yaml /app/

RUN mkdir -p /app/data /app/models /data/datasets /data/models \
    && useradd -m -u 1000 appuser && chown -R appuser:appuser /app /data
USER appuser

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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
