"""Generate synthetic training rows for weak subjects via the OpenAI API — a
standalone data tool, deliberately NOT part of the API.

Design against the known augmentation pitfalls:
- Few-shot conditioning on REAL curated rows of the subject (style anchor).
- Diversity forced per batch: rotating facets (school level, material type,
  sub-topic breadth) + rotating few-shot samples, so output does not collapse
  onto a few prototypes.
- Length constrained to the curated distribution (~100-500 chars of text).
- Output rows carry `generated_for=<label uri>` -- the mark the API reads
  (app/provenance.py), so they train but never validate. They were marked
  `source=synthetic`, which nothing reads, and went into validation and the test
  split like real rows (audit 2026-09-30, W03).

Usage (from api_v3/, OPENAI_API_KEY set):
    python scripts/generate_synthetic.py --curated data/data_30k.csv \
        --subjects 640,04006,04005,260,20005,510 --per-subject 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from enrich_tail import FILT, LABEL_COL, TEXT_COLS, combined_text  # noqa: E402

from app.data import clean_text, split_labels  # noqa: E402
from app.label_names import pair_names  # noqa: E402
from app.provenance import GENERATED_FOR  # noqa: E402

TITLE, DESC, KEYW = TEXT_COLS
# Rotating per batch: target audience/level + topic angle (breadth of the subject).
FACETS = [
    "Grundschule / einfache Sprache", "Sekundarstufe I", "Sekundarstufe II / Oberstufe",
    "berufliche Bildung / Ausbildung", "Erwachsenenbildung / Weiterbildung",
    "Randthemen und weniger typische Teilgebiete des Fachs",
    "fächerübergreifende Bezüge (aber das Fach bleibt der Kern)",
    "Alltagsbezug und aktuelle Anlässe", "Grundbegriffe und Einstieg", "Vertiefung für Fortgeschrittene",
]

PROMPT = """Kontext: Ein Katalog für BILDUNGSINHALTE (Lernmaterialien für Unterricht \
und Selbstlernen). Jeder Eintrag beschreibt EIN konkretes Lernmaterial zum Schulfach \
bzw. Fachgebiet "{subject}".

Echte Katalogeinträge zu diesem Fach als Stil- und Längenvorbild:
{examples}

Erzeuge {n} NEUE, DEUTLICH VERSCHIEDENE Katalogeinträge zu "{subject}".
Zielgruppe/Blickwinkel dieser Runde: {facet}.

Regeln:
- MISCHE die Materialtypen innerhalb deiner Antwort: Erklärvideo, Arbeitsblatt, Quiz, \
interaktive Übung, Webseite/Portal, Podcast, Unterrichtsentwurf, Lernspiel, Simulation, \
Präsentation — der Typ soll aus Titel oder Beschreibung hervorgehen.
- Decke unterschiedliche Teilgebiete von "{subject}" ab, nicht nur die typischsten Themen.
- Nutze fachtypische Begriffe natürlich im Text.
- description: 100-400 Zeichen, sachlich beschreibend wie die Beispiele.
- keywords: 3-6 Schlagwörter, kommagetrennt.
- KEINE Kopien oder Fast-Kopien der Beispiele.
Antworte NUR mit JSON: {{"items": [{{"title": "...", "description": "...", "keywords": "..."}}]}}"""


def openai_chat(model: str, prompt: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(json.load(resp)["choices"][0]["message"]["content"])


def real_examples(df: pd.DataFrame, uri: str, rng: random.Random, k: int = 4) -> list[str]:
    # The label exactly: a substring match took .../04003 rows as examples of .../040 (W03).
    rows = df[df[LABEL_COL].fillna("").map(lambda value: uri in split_labels(value, ","))]
    sample = rows.sample(min(k, len(rows)), random_state=rng.randint(0, 10**6))
    out = []
    for _, r in sample.iterrows():
        out.append(f"- Titel: {r[TITLE]!s:.80} | Beschreibung: {str(r[DESC])[:220]} "
                   f"| Schlagwörter: {str(r[KEYW])[:80]}")
    return out


def uri_names(curated: pd.DataFrame, vocabulary: Path) -> dict[str, str]:
    """URI -> display name: `label_names.json` when there is one (the only complete source),
    else what the CSV provably lines up (`pair_names`). Zipped by position, a name holding the
    separator shifted every later name onto the wrong URI, and the generator wrote rows for
    one subject under another's name (README, "Label display names"; audit W03)."""
    if vocabulary.exists():
        return dict(json.loads(vocabulary.read_text(encoding="utf-8")))
    names: dict[str, str] = {}
    for uris, shown in zip(curated[LABEL_COL].fillna(""), curated[f"{LABEL_COL}_DISPLAYNAME"].fillna(""),
                           strict=True):
        for uri, name in pair_names(split_labels(uris, ","), split_labels(shown, ",")):
            names.setdefault(uri, name)
    return names


def synthetic_row(item: dict, uri: str, name: str) -> dict:
    """One generated catalogue entry as a row of the curated CSV, marked as generated."""
    return {TITLE: item.get("title", ""), DESC: item.get("description", ""),
            KEYW: item.get("keywords", ""), LABEL_COL: uri,
            f"{LABEL_COL}_DISPLAYNAME": name, GENERATED_FOR: uri}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated", default="data/data_30k.csv")
    ap.add_argument("--subjects", required=True,
                    help="comma-separated discipline ids (e.g. 640,04006) or full URIs")
    ap.add_argument("--per-subject", type=int, default=200, help="target rows per subject (100-500)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/data_synthetic.csv")
    ap.add_argument("--names", default=None,
                    help="label_names.json (default: the one beside --curated, if any)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    curated = pd.read_csv(args.curated, sep=";", dtype=str, encoding="utf-8", low_memory=False)
    known = set(combined_text(curated, TEXT_COLS))

    uri_name = uri_names(curated, Path(args.names) if args.names
                         else Path(args.curated).parent / "label_names.json")

    subjects = [s if s.startswith("http") else FILT + s for s in args.subjects.split(",")]
    rows: list[dict] = []
    for uri in subjects:
        name = uri_name.get(uri, uri.rsplit("/", 1)[-1])
        made = 0
        batches = 0
        while made < args.per_subject and batches < args.per_subject:  # hard stop
            batches += 1
            facet = FACETS[(batches - 1) % len(FACETS)]
            prompt = PROMPT.format(subject=name, n=args.batch_size, facet=facet,
                                   examples="\n".join(real_examples(curated, uri, rng)))
            try:
                items = openai_chat(args.model, prompt).get("items", [])
            except Exception as exc:  # noqa: BLE001 - one failed batch must not kill the run
                print(f"  batch failed ({name}): {exc}")
                continue
            for it in items:
                text = clean_text(f"{it.get('title', '')} {it.get('description', '')} "
                                  f"{it.get('keywords', '')}")
                if not 100 <= len(text) <= 600 or text in known:
                    continue
                known.add(text)
                rows.append(synthetic_row(it, uri, name))
                made += 1
                if made >= args.per_subject:
                    break
            print(f"  {name}: {made}/{args.per_subject} (batch {batches}, {facet})")

    columns = [*curated.columns, *([GENERATED_FOR] if GENERATED_FOR not in curated.columns else [])]
    out = pd.DataFrame(rows).reindex(columns=columns, fill_value="")
    out.to_csv(args.out, sep=";", index=False, encoding="utf-8")
    print(f"\n{len(rows)} synthetische Zeilen -> {args.out}")


if __name__ == "__main__":
    main()
