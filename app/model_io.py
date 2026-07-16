"""Pickle-free model-bundle serialization and skops load-safety.

Split out of ``registry`` so the security-critical (de)serialization code — the
part that neutralises the pickle-RCE class on import — lives on its own, apart
from the cache/disk orchestration that consumes it. Kept minimal and dependency-
light for review.

A model bundle is a directory containing:
  - config.json      : backend kind, classes, thresholds, label map, task type
  - metrics.json     : training metadata + evaluation metrics
  - head.skops       : the sklearn OneVsRest(LogReg) head (skops, no pickle)
  - vectorizer.skops : (tfidf only) the fitted word/char vectorizers

Security: skops loads only known-safe types. Legitimate api_v3 bundles contain
zero "untrusted" types, so loading rejects any file that introduces one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from skops.io import dump as skops_dump
from skops.io import get_untrusted_types
from skops.io import load as skops_load

from .classifier import ClassifierModel
from .vectorizers import TfidfBackend

FORMAT_VERSION = 1
# Extra type names we explicitly trust beyond skops defaults. Empty: our own
# bundles produce no untrusted types, so anything extra is rejected.
_ALLOWED_EXTRA_TYPES: set[str] = set()


class UnsafeModelError(Exception):
    """Raised when a model file contains untrusted types, an unsafe layout, or is
    unreadable/corrupt (any bundle we cannot safely load)."""


def _safe_skops_load(path: Path):
    try:
        untrusted = get_untrusted_types(file=str(path))
    except Exception as exc:  # noqa: BLE001 - corrupt/non-skops container -> uniform load error
        raise UnsafeModelError(f"Cannot read {path.name}: {exc!r}") from exc
    disallowed = [t for t in untrusted if t not in _ALLOWED_EXTRA_TYPES]
    if disallowed:
        raise UnsafeModelError(f"Refusing to load {path.name}: untrusted types {disallowed}")
    # Trust ONLY the explicit allowlist, never the discovered `untrusted` set
    # (which is [] here anyway). Identical behaviour today, but stays safe if
    # _ALLOWED_EXTRA_TYPES is ever populated -- otherwise we would trust exactly
    # the types we just decided to allow, re-opening the RCE class this guards.
    try:
        return skops_load(str(path), trusted=sorted(_ALLOWED_EXTRA_TYPES))
    except Exception as exc:  # noqa: BLE001 - deserialization failure -> uniform load error
        raise UnsafeModelError(f"Cannot load {path.name}: {exc!r}") from exc


def _write_bundle(
    directory: Path,
    model: ClassifierModel,
    metadata: dict,
    on_step: Callable[[str], None] = lambda _msg: None,
) -> None:
    # on_step fires before each file: under memory pressure a single skops dump
    # can run for the better part of an hour, and these steps are the only
    # liveness signal (and the only hint WHICH file is crawling) during that.
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": FORMAT_VERSION,
        "backend_kind": model.vectorizer.kind,
        "classes": model.classes,
        "task_type": model.task_type,
        "avg_labels": model.avg_labels,
        "global_threshold": model.global_threshold,
        "per_label_thresholds": model.per_label_thresholds,
        "uri_to_label": model.uri_to_label,
    }
    on_step("Writing config.json + metrics.json")
    (directory / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    on_step("Writing head.skops")
    skops_dump(model.head, str(directory / "head.skops"))
    on_step("Writing vectorizer.skops")
    skops_dump(
        [model.vectorizer.word_vec, model.vectorizer.char_vec],
        str(directory / "vectorizer.skops"),
    )


def _read_bundle(directory: Path) -> tuple[ClassifierModel, dict]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.json"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    head = _safe_skops_load(directory / "head.skops")
    kind = config["backend_kind"]
    if kind != "tfidf":
        raise UnsafeModelError(f"Unsupported backend kind: {kind!r}")
    word_vec, char_vec = _safe_skops_load(directory / "vectorizer.skops")
    vectorizer = TfidfBackend()
    vectorizer.word_vec = word_vec
    vectorizer.char_vec = char_vec

    model = ClassifierModel(
        vectorizer=vectorizer,
        head=head,
        classes=config["classes"],
        task_type=config["task_type"],
        avg_labels=config["avg_labels"],
        uri_to_label=config.get("uri_to_label", {}),
        global_threshold=config["global_threshold"],
        per_label_thresholds=config.get("per_label_thresholds", {}),
    )
    return model, metadata
