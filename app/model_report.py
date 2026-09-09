"""Read-only reporting over a bundle's JSON documents: what a model is, where it is weak.

Split out of ``registry`` along the seam this project already uses once — ``data`` owns
loading and splitting, ``dataset_stats`` owns describing the result. The registry owns
*where and when* the disk is touched (cache, locks, atomic publish); this owns what the
two documents inside a bundle mean to a reader, which is a different reason to change
and has changed twice as often.

The functions take the documents rather than a model name, so they stay pure, testable
without a disk, and — the reason the split came when it did — so both reports can be
served from ONE read: ``label_diagnostics`` used to re-parse ``config.json`` after
``info`` had already parsed it, outside the same lock, which let a concurrent delete
land between the two halves of one answer.

Every value read here is foreign input; see ``bundle_meta`` for why it degrades
instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bundle_meta import as_count, as_mapping, as_names, per_label_f1
from .label_names import label_vocabulary


def read_documents(directory: Path) -> tuple[dict, dict]:
    """A bundle's ``config.json`` and ``metrics.json``, read together.

    Metrics are optional: bundles trained before a field existed — or before the file
    did — must still be describable, so an absent one reads as an empty document.
    """
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.json"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return config, metadata


def describe(name: str, config: dict, metadata: dict) -> dict:
    """The model overview: its config, its metrics, and the vocabulary it labels from."""
    # uri_to_label holds one entry per label and no overview reads it.
    overview = {key: value for key, value in config.items() if key != "uri_to_label"}
    # Derived, never stored: it follows from the labels, so it stays correct for
    # bundles trained before this existed and cannot be typed in wrong.
    vocabulary = label_vocabulary(config.get("classes") or [])
    return {"name": name, **overview, "label_vocabulary": vocabulary, "metadata": metadata}


def label_diagnostics(config: dict, metadata: dict) -> list[dict]:
    """Per label: its F1, how many rows carry it, and the threshold serving applies.

    Weakest first — the end anyone reviewing a model looks at. A model's headline F1
    says how good it is on average; this says *where* it is weak, which is what decides
    whether a given answer deserves a second look.

    ``f1`` and ``support`` are ``None`` for bundles trained before they were recorded.
    ``threshold`` is ``None`` for binary/multiclass, where serving picks the argmax and
    never reads a threshold — reporting one would describe a rule the model does not apply.
    """
    scores = per_label_f1(metadata)
    support = as_mapping(metadata.get("per_label_support"))
    thresholds = as_mapping(config.get("per_label_thresholds"))
    names = as_mapping(config.get("uri_to_label"))
    single_label = config.get("task_type") in ("binary", "multiclass")

    entries = [
        {
            "uri": uri,
            "label": names.get(uri, uri),
            "f1": scores.get(uri),
            "support": as_count(support.get(uri)),
            "threshold": None if single_label else thresholds.get(uri, config.get("global_threshold")),
        }
        for uri in as_names(config.get("classes"))
    ]
    # Unscored labels last: unknown is not the same as weak.
    return sorted(entries, key=lambda entry: (entry["f1"] is None, entry["f1"] or 0.0))
