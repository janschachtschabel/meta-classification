"""Repair a trained bundle's display names in place -- no retraining needed.

``uri_to_label`` is plain JSON inside a bundle's ``config.json`` and is used for
presentation only: the classifier keys everything off label URIs. So a bundle whose names
were mangled by the comma-in-display-name export defect (see ``app/label_names.pair_names``) can
be corrected after the fact.

Only names are touched. ``classes``, thresholds and both skops members stay byte-identical,
so predictions before and after are the same -- verified by comparing the class list.

Usage (from the repo root), dry run first:
    python scripts/patch_bundle_labels.py --names data/label_names.json
    python scripts/patch_bundle_labels.py --names data/label_names.json --apply
    python scripts/patch_bundle_labels.py --model faecher_300k_auto --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def patch(bundle: Path, names: dict[str, str], *, apply: bool) -> tuple[int, int, int]:
    """Return (corrected, added, still_nameless) for one bundle."""
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classes: list[str] = config["classes"]
    current: dict[str, str] = config.get("uri_to_label", {})

    # Narrowed to this bundle's own labels: a bundle carries its vocabulary, not the world's.
    updated = dict(current)
    corrected = added = 0
    for uri in classes:
        authoritative = names.get(uri)
        if not authoritative:
            continue
        if uri not in current:
            added += 1
        elif current[uri] != authoritative:
            corrected += 1
        else:
            continue
        updated[uri] = authoritative
    nameless = [uri for uri in classes if not updated.get(uri)]

    print(f"\n=== {bundle.name} ({len(classes)} labels) ===")
    print(f"  names corrected (were WRONG): {corrected}")
    print(f"  names added (were missing)  : {added}")
    print(f"  still without a name        : {len(nameless)}  {[u or '<empty>' for u in nameless[:5]]}")
    for uri in classes:
        authoritative = names.get(uri)
        if authoritative and uri in current and current[uri] != authoritative:
            print(f"    {uri.rsplit('/', 1)[-1]:>10}  {current[uri]!r}  ->  {authoritative!r}")

    if apply and (corrected or added):
        backup = config_path.with_suffix(".json.bak")
        if not backup.exists():  # keep the FIRST original, not the previous patch
            shutil.copy2(config_path, backup)
        config["uri_to_label"] = updated
        # Same writer settings as model_io._write_bundle, so the file stays comparable.
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Re-read and assert the parts that must NOT have changed.
        after = json.loads(config_path.read_text(encoding="utf-8"))
        assert after["classes"] == classes, "class list changed - aborting"
        assert after["per_label_thresholds"] == config["per_label_thresholds"], "thresholds changed"
        print(f"  written (backup: {backup.name}); classes + thresholds unchanged")
    elif apply:
        print("  nothing to change")
    else:
        print("  DRY RUN - pass --apply to write")
    return corrected, added, len(nameless)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", default=str(_REPO_ROOT / "data" / "label_names.json"))
    parser.add_argument("--models-dir", default=str(_REPO_ROOT / "models"))
    parser.add_argument("--model", action="append", default=None, help="bundle name (repeatable)")
    parser.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    args = parser.parse_args()

    names: dict[str, str] = json.loads(Path(args.names).read_text(encoding="utf-8"))
    models_dir = Path(args.models_dir)
    targets = (
        [models_dir / name for name in args.model]
        if args.model
        else sorted(p for p in models_dir.iterdir() if (p / "config.json").exists())
    )
    print(f"{len(names)} authoritative names from {args.names}")
    for bundle in targets:
        if not (bundle / "config.json").exists():
            print(f"\n=== {bundle.name}: no config.json, skipped ===")
            continue
        patch(bundle, names, apply=args.apply)


if __name__ == "__main__":
    main()
