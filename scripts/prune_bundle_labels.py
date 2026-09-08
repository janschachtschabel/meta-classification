"""Remove container labels from a trained bundle -- no retraining needed.

A label value ending in ``/`` names a namespace, not a concept (see
``app.data.is_container_label``). Bundles trained before that guard existed carry one as an
ordinary class: measured, ``…/vocabs/discipline/`` sat on 522 rows at F1 0.4096.

This drops the class from the OneVsRest head. In multilabel mode ``predict_proba`` stacks
exactly one column per entry in ``estimators_``, so removing an entry removes its column
and leaves the others untouched -- which the script PROVES by comparing probabilities for
the kept labels before and after, rather than assuming it.

Metrics honesty: ``f1_macro`` is the unweighted mean of the per-label F1s, so it is
recomputed exactly. ``f1_micro`` / ``precision_macro`` / ``recall_macro`` cannot be derived
without the original predictions, so they are moved aside under ``metrics_before_pruning``
instead of being left in place looking current.

Usage (from the repo root), dry run first:
    python scripts/prune_bundle_labels.py
    python scripts/prune_bundle_labels.py --apply
    python scripts/prune_bundle_labels.py --model faecher_300k_auto --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.data import is_container_label  # noqa: E402
from app.model_io import UnsafeModelError  # noqa: E402
from app.registry import _BACKUP_SUFFIX, Registry  # noqa: E402

# Texts only need to exercise the vectorizer; the comparison is per-column, so any
# non-degenerate batch proves the kept columns are unchanged.
_PROBE = [
    "Übungsblatt zur Integralrechnung mit Lösungen für die Oberstufe",
    "Vorlesungsskript Thermodynamik und Strömungsmechanik für Maschinenbau",
    "",
    "Arbeitsblatt Grammatik und Rechtschreibung Klasse 5",
]


def prune(registry: Registry, name: str, *, apply: bool) -> bool:
    try:
        model, metadata = registry.load_fresh(name)
    except UnsafeModelError as exc:
        # Older bundles (e.g. format 1) cannot be opened by this version at all. Report and
        # move on: one unreadable bundle must not stop the others from being repaired.
        print(f"\n=== {name}: skipped — {exc} ===")
        return False
    doomed = [i for i, uri in enumerate(model.classes) if is_container_label(uri)]
    print(f"\n=== {name}: {len(model.classes)} classes ===")
    if not doomed:
        print("  no container labels — nothing to do")
        return False

    per_label = dict((metadata.get("metrics") or {}).get("per_label_f1") or {})
    for i in doomed:
        uri = model.classes[i]
        print(f"  removing {uri!r}")
        print(f"    shown as {model.uri_to_label.get(uri)!r}  "
              f"F1={per_label.get(uri)}  threshold={model.per_label_thresholds.get(uri)}")

    keep = [i for i in range(len(model.classes)) if i not in set(doomed)]
    before = model.predict_proba(_PROBE)[:, keep]

    gone = [model.classes[i] for i in doomed]
    model.classes = [model.classes[i] for i in keep]
    model.head.estimators_ = [model.head.estimators_[i] for i in keep]
    # Keep the wrapper self-consistent: these are renumbered class INDICES, not URIs.
    model.head.classes_ = np.arange(len(keep))
    if getattr(model.head, "label_binarizer_", None) is not None:
        model.head.label_binarizer_.classes_ = np.arange(len(keep))
    for uri in gone:
        model.per_label_thresholds.pop(uri, None)
        model.uri_to_label.pop(uri, None)
        model.per_label_f1.pop(uri, None)
        per_label.pop(uri, None)

    after = model.predict_proba(_PROBE)
    if after.shape != before.shape or not np.array_equal(after, before):
        raise SystemExit(f"{name}: probabilities for KEPT labels changed — aborting, nothing written")
    print(f"  kept-label probabilities bit-identical ({after.shape[1]} classes remain)")

    metrics = dict(metadata.get("metrics") or {})
    old_macro = metrics.get("f1_macro")
    metrics["per_label_f1"] = per_label
    metrics["n_labels"] = len(keep)
    if per_label:
        metrics["f1_macro"] = float(np.mean(list(per_label.values())))
        print(f"  f1_macro recomputed: {old_macro:.4f} -> {metrics['f1_macro']:.4f}")
    # These depend on the removed label's TP/FP/FN and cannot be recomputed here.
    stale = {k: metrics.pop(k) for k in ("f1_micro", "precision_macro", "recall_macro") if k in metrics}
    if stale:
        metrics["metrics_before_pruning"] = stale
        print(f"  moved aside (not recomputable): {', '.join(stale)}")
    metadata["metrics"] = metrics
    metadata["n_labels"] = len(keep)
    metadata["pruned_container_labels"] = gone

    if not apply:
        print("  DRY RUN — pass --apply to write")
        return False

    # The suffix comes from the registry, which is what excludes the copy from
    # list() — a backup named anything else would be served as a model.
    backup = registry.dir / f"{name}{_BACKUP_SUFFIX}"
    if not backup.exists():
        shutil.copytree(registry.dir / name, backup)
    registry.save(name, model, metadata)
    print(f"  written (backup: {backup.name})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(_REPO_ROOT / "models"))
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    registry = Registry(args.models_dir, 1)
    for name in args.model or registry.list():
        prune(registry, name, apply=args.apply)


if __name__ == "__main__":
    main()
