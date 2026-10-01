"""What the process does on the way up and on the way down.

Split out of `main.py`, which had four reasons to change in it. This one is "what happens at
boot and at shutdown": the configuration check that refuses an unusable deployment, the sweeps
that clean up after a kill, the optional warmups, and the lifespan that orders them.

Only the two warmups are best-effort — they move a cost and nothing depends on them, so they
log and carry on. Everything else in `lifespan` is load-bearing and deliberately unguarded: the
auth check refuses a deployment nobody could administer, and `ensure_dirs` and the sweeps have to
succeed for the process to have somewhere to write. A failure in any of those stops the boot,
which is the intended outcome — coming up without writable storage only defers the error to the
first request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .jobs import job_runner
from .metadata import generate
from .registry import Registry, get_registry
from .settings import Settings, get_settings

logger = logging.getLogger("api_v3")


def _warmup_models(registry: Registry, names: list[str]) -> None:
    """Preload configured models into the LRU cache and run one empty-text
    prediction, so the first real /predict pays no cold skops-load. Best-effort:
    a missing or unreadable model is logged and skipped, never fatal to startup."""
    for name in names:
        try:
            registry.get(name).baseline_proba()
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort; never fail startup
            logger.warning("Warmup skipped model %r: %r", name, exc)
        else:
            logger.info("Warmed up model %r", name)


def _warmup_metadata() -> None:
    """Run the metadata generators once so the first /metadata request pays no cold load.

    `wordfreq` loads its German frequency table on the first lookup rather than at import
    (measured 245-286 ms over four fresh interpreters, ~58 MB resident), and pysbd compiles its
    segmentation regexes on the first split (~20 ms; the Segmenter object itself is built at
    import). One real generation touches both, plus the stemmer. Best-effort like
    `_warmup_models`: this only moves a cost, so a failure here must never stop the process
    from serving.
    """
    text = ("Photosynthese\n"
            "Pflanzen wandeln Licht, Wasser und Kohlendioxid in Zucker um. "
            "Dabei entsteht Sauerstoff, den Menschen und Tiere zum Atmen brauchen.")
    try:
        generate(text)
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort; never fail startup
        logger.warning("Metadata warmup skipped: %r", exc)
    else:
        logger.info("Warmed up the descriptive-metadata generators")


def sweep_upload_staging(data_dir: Path) -> int:
    """Delete upload staging orphaned by a kill; returns how many were removed.

    A dataset upload spools to ``<name>.part`` and a CSV classification to a hidden
    ``.predict-*.csv.tmp``, both renamed or deleted on every normal and error path. A
    SIGKILL mid-stream leaves one behind, and neither carries a dataset suffix, so the
    listings never show it and nothing reclaims the disk. The models dir has been swept
    for exactly this since the export staging landed (``Registry.sweep_stale_tmp``); this
    is the same problem in the other directory.
    """
    if not data_dir.exists():
        return 0
    removed = 0
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".part") or (path.name.startswith(".") and path.name.endswith(".tmp")):
            path.unlink(missing_ok=True)
            removed += not path.exists()
    return removed


# The prefix of the keys `.env.example` ships: published, so they protect nothing.
_PLACEHOLDER_PREFIX = "change-me"
_MAKE_A_KEY = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'


def _check_auth_configuration(settings: Settings) -> None:
    """Refuse to start an authenticated deployment that nobody can administer -- or that
    anyone could.

    With auth on and no admin key, ``_role_for_key`` can never return "admin": every
    request 401s and no model can ever be trained, imported or deleted. That reads
    like a client-side key problem and has cost real debugging time, so fail here
    with the variable name instead. A readonly-only deployment is legitimate, so
    only the admin key is required.

    A key that is set must also be usable and secret (audit 2026-09-30, S10). Outside ASCII
    it made every request carrying a key a 500 (``compare_digest`` refuses non-ASCII text),
    and it could not be matched reliably anyway: clients encode such a header differently.
    And the ``.env.example`` placeholders are known to anyone who has read the repository.
    The messages name the variable, never the value.
    """
    if not settings.auth_enabled:
        return
    if not settings.api_key_admin:
        raise RuntimeError(
            "APIV3_AUTH_ENABLED is true but APIV3_API_KEY_ADMIN is not set — every "
            "request would be rejected with 401. Set the key, or run with "
            "APIV3_AUTH_ENABLED=false for local use."
        )
    for variable, key in (("APIV3_API_KEY_ADMIN", settings.api_key_admin),
                          ("APIV3_API_KEY_READONLY", settings.api_key_readonly)):
        if not key:
            continue
        if not key.isascii():
            raise RuntimeError(
                f"{variable} contains characters outside ASCII. A key travels in an HTTP "
                "header, where clients encode those differently, so it could never be "
                f"matched reliably. Use a random ASCII key: {_MAKE_A_KEY}"
            )
        if key.startswith(_PLACEHOLDER_PREFIX):
            raise RuntimeError(
                f"{variable} is still the placeholder from .env.example, which anyone who "
                f"has read the repository knows. Set a key of your own: {_MAKE_A_KEY}"
            )
    if settings.api_key_admin == settings.api_key_readonly:
        logger.warning(
            "APIV3_API_KEY_ADMIN and APIV3_API_KEY_READONLY are identical — the "
            "readonly role grants full admin access."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create storage dirs on startup, sweep model staging dirs orphaned by a
    crashed/killed save (a hidden ``.name.tmp`` whose atomic rename never ran),
    and preload any configured warmup models."""
    settings = get_settings()
    _check_auth_configuration(settings)
    settings.ensure_dirs()
    registry = get_registry()
    swept = registry.sweep_stale_tmp()
    if swept:
        logger.warning("Swept %d orphaned model staging dir(s) from a previous crash", swept)
    spooled = sweep_upload_staging(settings.data_dir)
    if spooled:
        logger.warning("Swept %d orphaned upload staging file(s) from a previous crash", spooled)
    if settings.warmup_models_list:
        resident = settings.effective_max_models_in_memory()
        if resident > settings.max_models_in_memory:
            # Never silently: the operator set a RAM ceiling and the warmup list
            # raised it, so the extra memory has to be visible in the log.
            logger.info(
                "Model cache holds %d models (raised from APIV3_MAX_MODELS_IN_MEMORY=%d "
                "to fit the %d warmup models)",
                resident, settings.max_models_in_memory, len(settings.warmup_models_list),
            )
        _warmup_models(registry, settings.warmup_models_list)
    if settings.warmup_metadata:
        _warmup_metadata()
    logger.info(
        "api_v3 ready (models_dir=%s, auth=%s)", settings.models_dir, settings.auth_enabled
    )
    yield
    # Ask a running training to stop at its next checkpoint (between the C fits,
    # before the deploy fit). The thread is a daemon and dies with the process
    # anyway; this gives it the chance to end cleanly inside the termination grace
    # period instead, which is what the Helm chart's 60 s already assumed.
    job_runner.stop()
    logger.info("api_v3 shutting down")


