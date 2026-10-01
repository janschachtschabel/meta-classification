"""Build-time tool: download SKOS vocabularies and write a URI -> label lookup.

Why this exists: the WLO CSV export separates label URIs with commas AND their display
names with commas, so a name that itself contains a comma ("Rechts-, Wirtschafts- und
Sozialwissenschaften") makes the two lists misalign. Names recovered from the CSV are
therefore incomplete -- this file supplies the authoritative ones.

Deliberately a SCRIPT, not runtime code. `app/` never fetches a URL (that is the
SSRF boundary the project keeps); it only reads the JSON file this produces.

Usage (from the repo root):
    python scripts/fetch_vocab_labels.py
    python scripts/fetch_vocab_labels.py --out data/label_names.json --lang de
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

# SKOS ConceptScheme documents. Add a line to cover another vocabulary.
VOCABULARIES = (
    "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json",
    "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/hochschulfaechersystematik/index.json",
)
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Fewer than this from one vocabulary is not a vocabulary: a scheme without `hasTopConcept`
# gave `{}`, and `{}` was written over a good label_names.json (audit 2026-09-30, W07).
MIN_LABELS = 10


def _walk(concepts: object, lang: str, out: dict[str, str]) -> None:
    """Collect id -> prefLabel[lang] from a SKOS tree (`narrower` nests arbitrarily deep)."""
    if not isinstance(concepts, list):
        return
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        uri = concept.get("id")
        labels = concept.get("prefLabel")
        if isinstance(uri, str) and isinstance(labels, dict):
            # Fall back to any language rather than dropping the concept: a missing German
            # label is still better identified by its English one than by a bare URI.
            name = labels.get(lang) or next(
                (v for v in labels.values() if isinstance(v, str) and v), None
            )
            if isinstance(name, str) and name.strip():
                out[uri] = name.strip()
        _walk(concept.get("narrower"), lang, out)


def fetch(url: str, lang: str) -> dict[str, str]:
    # https only, overrides included: `--url file:///...` was read as readily (W07).
    if not url.startswith("https://"):
        raise SystemExit(f"refusing {url!r}: vocabularies are fetched over https only")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - https only, checked above
        scheme = json.loads(response.read().decode("utf-8"))
    names: dict[str, str] = {}
    _walk(scheme.get("hasTopConcept"), lang, names)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_REPO_ROOT / "data" / "label_names.json"))
    parser.add_argument("--lang", default="de", help="preferred prefLabel language")
    parser.add_argument("--url", action="append", default=None, help="override vocabulary URLs")
    args = parser.parse_args()

    merged: dict[str, str] = {}
    for url in args.url or VOCABULARIES:
        names = fetch(url, args.lang)
        print(f"{len(names):>5} labels from {url}")
        if len(names) < MIN_LABELS:
            raise SystemExit(f"{url} gave {len(names)} labels (fewer than {MIN_LABELS}): not a "
                             "concept scheme as expected -- nothing written")
        overlap = set(names) & set(merged)
        if overlap:
            print(f"      note: {len(overlap)} URIs already seen; keeping the first")
        merged = {**names, **merged}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Whole or not at all: written in place, an interrupted write left half a JSON document
    # where the names were (W07). A sibling temp file, then one rename.
    handle, temp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(sorted(merged.items())), ensure_ascii=False, indent=1))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, out)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    print(f"\n{len(merged)} labels -> {out}")


if __name__ == "__main__":
    main()
