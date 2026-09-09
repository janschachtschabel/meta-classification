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

import copy
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from skops.io import dump as skops_dump
from skops.io import get_untrusted_types
from skops.io import load as skops_load

from . import __version__
from .bundle_meta import per_label_f1
from .classifier import ClassifierModel
from .label_names import is_container_label
from .vectorizers import TfidfBackend

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
# Extra type names we explicitly trust beyond skops defaults. Empty: our own
# bundles produce no untrusted types, so anything extra is rejected.
_ALLOWED_EXTRA_TYPES: set[str] = set()
# The TF-IDF vocabularies live here instead of inside vectorizer.skops. skops walks a
# dict entry by entry and is super-quadratic in the count: 64k terms measured at 179 s
# and 45 MB, 200k terms at ~45 min and 148 MB — while the 48 MB float head writes in a
# second. As JSON the same data is <0.1 s and ~1 MB. Format 2 and later only.
_VOCAB_FILE = "vocabulary.json"

# Transport-only members: generated per export, verified on import, never kept in the
# installed bundle (a stale copy on disk would be zipped alongside the fresh one).
MANIFEST_FILE = "manifest.json"
CARD_FILE = "README.md"


class UnsafeModelError(Exception):
    """Raised when a model file contains untrusted types, an unsafe layout, or is
    unreadable/corrupt (any bundle we cannot safely load)."""


def build_manifest(name: str, members: dict[str, bytes], metadata: dict) -> dict:
    """Describe an archive: a SHA-256 for every member, plus what it is.

    Deliberately built at EXPORT rather than at save time. ``metrics.json`` is
    mutable by design (``PUT /models/{name}/info``) and ``config.json`` is rewritten
    by the label-repair scripts, so a manifest stored next to them would be
    invalidated by every legitimate edit — and a load-time check would then refuse a
    perfectly good bundle. What actually needs protecting is the 50-180 MB download
    between two servers, and that is exactly the export/import boundary.
    """
    metrics = metadata.get("metrics") or {}
    return {
        "model_name": name,
        "app_version": __version__,
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "created_at": metadata.get("created_at"),
        "n_labels": metadata.get("n_labels"),
        "f1_macro": metrics.get("f1_macro"),
        "dataset": metadata.get("dataset"),
        "files": {
            member: hashlib.sha256(payload).hexdigest()
            for member, payload in sorted(members.items())
        },
    }


def verify_manifest(manifest: object, members: dict[str, bytes]) -> None:
    """Check every member against the manifest; raise ``UnsafeModelError`` on any drift.

    Covers three failures with one comparison: a truncated download, a member altered
    in transit, and a member the manifest does not mention at all.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise UnsafeModelError(f"{MANIFEST_FILE} is malformed")
    expected: dict = manifest["files"]
    for member, payload in sorted(members.items()):
        digest = expected.get(member)
        if digest is None:
            raise UnsafeModelError(f"{member} is not listed in {MANIFEST_FILE}")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise UnsafeModelError(
                f"{member} does not match its checksum in {MANIFEST_FILE} "
                "(the archive was altered or arrived incomplete)"
            )
    missing = sorted(set(expected) - set(members))
    if missing:
        raise UnsafeModelError(f"{MANIFEST_FILE} lists files the archive lacks: {missing}")


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
    lean, vocabularies = _split_vocabularies(model.vectorizer)
    skops_dump(lean, str(directory / "vectorizer.skops"))
    (directory / _VOCAB_FILE).write_text(
        json.dumps(vocabularies, ensure_ascii=False), encoding="utf-8"
    )


def _split_vocabularies(vectorizer: TfidfBackend) -> tuple[list, list[list[str]]]:
    """Sub-vectorizers without ``vocabulary_``, plus their terms in column order.

    The copies are SHALLOW: the numpy arrays are shared (so this costs nothing) and the
    live object is left intact — it may already be serving requests from the LRU cache,
    and stripping an attribute off it would break prediction mid-flight.
    """
    lean: list = []
    vocabularies: list[list[str]] = []
    for sub in (vectorizer.word_vec, vectorizer.char_vec):
        if sub is None:  # word-only profiles have no char vectorizer
            lean.append(None)
            vocabularies.append([])
            continue
        without_vocabulary = copy.copy(sub)
        del without_vocabulary.vocabulary_
        lean.append(without_vocabulary)
        vocabularies.append(sorted(sub.vocabulary_, key=sub.vocabulary_.get))
    return lean, vocabularies


def _attach_vocabularies(sub_vectorizers: list, vocabularies: object) -> None:
    """Rebuild ``vocabulary_`` from the JSON member, validated against the model.

    Bundles are importable, so this is a trust boundary. A vocabulary of the wrong
    length would silently produce a feature matrix of the wrong width and only fail
    deep inside the head; duplicate terms would collapse columns. Reject both here.
    """
    if not isinstance(vocabularies, list) or len(vocabularies) != len(sub_vectorizers):
        raise UnsafeModelError(f"{_VOCAB_FILE} does not describe this bundle's vectorizers")
    for sub, terms in zip(sub_vectorizers, vocabularies, strict=True):
        if sub is None:
            continue
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise UnsafeModelError(f"{_VOCAB_FILE} must hold a list of term lists")
        if len(set(terms)) != len(terms):
            raise UnsafeModelError(f"{_VOCAB_FILE} contains duplicate terms")
        expected = len(sub.idf_)
        if len(terms) != expected:
            raise UnsafeModelError(
                f"{_VOCAB_FILE} has {len(terms)} terms but the model expects {expected}"
            )
        sub.vocabulary_ = {term: index for index, term in enumerate(terms)}


def _read_bundle(directory: Path) -> tuple[ClassifierModel, dict]:
    # Any parse/shape failure below means "a bundle we cannot safely load" —
    # map it to UnsafeModelError so routes answer 422/400 instead of a 500
    # (corrupt config.json, missing keys, wrong-shaped vectorizer container).
    # FileNotFoundError propagates unchanged (routes map it to 404/TOCTOU).
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        metrics_path = directory / "metrics.json"
        metadata = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        head = _safe_skops_load(directory / "head.skops")
        kind = config["backend_kind"]
        if kind != "tfidf":
            raise UnsafeModelError(f"Unsupported backend kind: {kind!r}")
        vocabulary_path = directory / _VOCAB_FILE
        if not vocabulary_path.exists():
            # Format 1 kept the vocabularies inside vectorizer.skops. Say so plainly:
            # without this the missing file surfaces as a bare FileNotFoundError, which
            # the routes map to "model not found" — a misleading 404 for a bundle that
            # is present but simply predates format 2.
            raise UnsafeModelError(
                f"{directory.name!r} is a format-1 bundle (no {_VOCAB_FILE}); "
                f"format {FORMAT_VERSION} moved the vocabulary out of the skops "
                "container. Retrain the model to use it with this version."
            )
        word_vec, char_vec = _safe_skops_load(directory / "vectorizer.skops")
        _attach_vocabularies(
            [word_vec, char_vec],
            json.loads(vocabulary_path.read_text(encoding="utf-8")),
        )
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
            per_label_f1=per_label_f1(metadata),
        )
        _warn_about_container_labels(directory.name, model.classes)
    except (AttributeError, ValueError, KeyError, TypeError) as exc:
        raise UnsafeModelError(f"Invalid model bundle in {directory.name!r}: {exc!r}") from exc
    return model, metadata


def _warn_about_container_labels(name: str, classes: list[str]) -> None:
    """Flag classes that name a namespace rather than a concept (see ``data.split_labels``).

    Training cannot produce these any more, but an *import* can: a bundle built by an older
    version carries its own ``classes``. Repairing it here is not possible — the class list
    is positionally tied to the head's estimators, so dropping an entry without dropping the
    matching estimator would silently shift every probability onto the wrong label. Hence a
    loud warning plus the name of the script that does it properly, rather than a refusal
    (the bundle is otherwise perfectly usable) or a silent pass.
    """
    offenders = [uri for uri in classes if is_container_label(uri)]
    if offenders:
        logger.warning(
            "Model %r has %d container label(s) that name a namespace, not a concept: %s. "
            "They were trained as ordinary classes and can be predicted. Repair with "
            "`python scripts/prune_bundle_labels.py --model %s --apply`.",
            name, len(offenders), ", ".join(repr(u) for u in offenders), name,
        )
