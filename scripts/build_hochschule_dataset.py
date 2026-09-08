"""Combine staging Hochschule content with the 300k export into one higher-ed dataset.

Steps, in order, each counted so the losses stay visible:
  1. staging JSONL -> keep rows whose ccm:educationalcontext really holds the hochschule
     URI (the sweep filter was the imprecise short form on purpose) and whose top-level
     isPublic is not false.
  2. subject labels come from BOTH ccm:oeh_taxonid_university and ccm:taxonid, narrowed to
     the hochschulfaechersystematik vocabulary — measured on staging, higher-ed nodes use
     either field depending on the source.
  3. the 300k export -> rows with a hochschulfaechersystematik taxonid.
  4. CROSS-REPO dedupe. Node ids are useless here (different repository), so the keys are
     the normalised wwwurl and the normalised title, per instruction. Reports what each
     key contributed, and how risky title-only matching is, instead of assuming.

Writes a CSV with the column names api_v3 already trains on, so the existing request works
unchanged: the merged labels go into properties.ccm:taxonid.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

csv.field_size_limit(10_000_000)
_DATA = Path(__file__).resolve().parent.parent / "data"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jsonl", default=str(_DATA / "staging_hochschule_raw.jsonl"),
                    help="raw edu-sharing nodes from scripts in wlo-content-downloader")
parser.add_argument("--export", default=str(_DATA / "data_300k.csv"),
                    help="the existing WLO CSV export to merge with")
parser.add_argument("--out", default=str(_DATA / "data_hochschule_combined.csv"))
_args = parser.parse_args()
JSONL, EXPORT, OUT = Path(_args.jsonl), Path(_args.export), Path(_args.out)

HS_CONTEXT = "http://w3id.org/openeduhub/vocabs/educationalContext/hochschule"
UNI_VOCAB = "hochschulfaechersystematik"
TITLE = "properties.cclom:title"
DESC = "properties.cclom:general_description"
KEYWORDS = "properties.cclom:general_keyword"
LABELS, LABEL_NAMES = "properties.ccm:taxonid", "properties.ccm:taxonid_DISPLAYNAME"
WWWURL, SOURCE, ORIGIN = "properties.ccm:wwwurl", "properties.ccm:replicationsource", "origin"
COLUMNS = [TITLE, DESC, KEYWORDS, LABELS, LABEL_NAMES, WWWURL, SOURCE, ORIGIN]

_WS = re.compile(r"\s+")


def first(value: object) -> str:
    """edu-sharing returns every property as a list; the 300k export as a plain string."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def joined(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return "" if value is None else str(value)


def norm_title(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def norm_url(text: str) -> str:
    """Scheme, www and a trailing slash are not identity — the rest is."""
    url = text.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/")


def uni_labels(*cells: object) -> list[str]:
    """Higher-education label URIs from any of the source fields.

    Splits on BOTH ``,`` and ``|`` so a CSV written with either multi-value separator works
    unchanged. That is safe precisely here: label URIs never contain either character — it is
    only the human-readable *names* that do, which is what made the comma ambiguous in the
    first place.
    """
    out: list[str] = []
    for cell in cells:
        values = cell if isinstance(cell, list) else re.split(r"[,|]", str(cell or ""))
        for value in values:
            uri = str(value).strip()
            if UNI_VOCAB in uri and not uri.endswith("/") and uri not in out:
                out.append(uri)
    return out


# --- 1..2 staging ---------------------------------------------------------------------
staging: list[dict] = []
stats: Counter[str] = Counter()
with open(JSONL, encoding="utf-8") as handle:
    for line in handle:
        stats["read"] += 1
        node = json.loads(line)
        props = node.get("properties") or {}
        if HS_CONTEXT not in (props.get("ccm:educationalcontext") or []):
            stats["dropped: context not hochschule"] += 1
            continue
        if node.get("isPublic") is False:
            stats["dropped: isPublic false"] += 1
            continue
        labels = uni_labels(props.get("ccm:oeh_taxonid_university"), props.get("ccm:taxonid"))
        if not labels:
            stats["dropped: no higher-ed subject"] += 1
            continue
        names = joined(props.get("ccm:oeh_taxonid_university_DISPLAYNAME")) or joined(
            props.get("ccm:taxonid_DISPLAYNAME"))
        staging.append({
            TITLE: first(props.get("cclom:title")),
            DESC: first(props.get("cclom:general_description")),
            KEYWORDS: joined(props.get("cclom:general_keyword")),
            LABELS: ", ".join(labels),
            LABEL_NAMES: names,
            WWWURL: first(props.get("ccm:wwwurl")),
            SOURCE: first(props.get("ccm:replicationsource")),
            ORIGIN: "staging",
        })
        stats["kept"] += 1

print(f"staging {JSONL.name}:")
for key, count in stats.most_common():
    print(f"  {key:<34} {count:>8,}")

# --- 3 the 300k export ----------------------------------------------------------------
export: list[dict] = []
with open(EXPORT, encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter=";"):
        labels = uni_labels(row.get(LABELS), row.get("properties.ccm:oeh_taxonid_university"))
        if not labels:
            continue
        export.append({
            TITLE: row.get(TITLE) or "",
            DESC: row.get(DESC) or "",
            KEYWORDS: row.get(KEYWORDS) or "",
            LABELS: ", ".join(labels),
            LABEL_NAMES: row.get(LABEL_NAMES) or "",
            WWWURL: row.get(WWWURL) or "",
            SOURCE: row.get(SOURCE) or "",
            ORIGIN: "export300k",
        })
print(f"\n300k export: {len(export):,} rows with a higher-ed subject")

# --- 4 cross-repo dedupe --------------------------------------------------------------
# The URL is identity WHEN PRESENT; the title is only a fallback for rows without one.
# 🟢 Measured why the title must not be a key in its own right: 32,164 titles map to more
# than one distinct URL, and treating title equality as identity discarded 34,886 genuinely
# different resources — 'quiz1.pdf' alone covers 28 distinct URLs, 'final exam' 28,
# 'introduction to psychology' 23. Only 86 of 159,316 rows lack a URL, so the fallback is
# nearly never needed anyway.
combined: list[dict] = []
seen_urls: set[str] = set()
seen_titles: set[str] = set()
merge: Counter[str] = Counter()
# The export goes first: it is the curated, already-trained-on side, so where the two
# overlap the established row wins and staging only ADDS.
for row in export + staging:
    url = norm_url(row[WWWURL])
    title = norm_title(row[TITLE])
    if url:
        if url in seen_urls:
            merge[f"duplicate by wwwurl ({row[ORIGIN]})"] += 1
            continue
        seen_urls.add(url)
    elif title:
        if title in seen_titles:
            merge[f"duplicate by title, no url ({row[ORIGIN]})"] += 1
            continue
        seen_titles.add(title)
    combined.append(row)

print("\ndedupe:")
for key, count in merge.most_common():
    print(f"  {key:<40} {count:>8,}")
print(f"  {'kept':<40} {len(combined):>8,}")

with open(OUT, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";")
    writer.writeheader()
    writer.writerows(combined)

origins = Counter(r[ORIGIN] for r in combined)
sources = Counter(r[SOURCE] or "(none)" for r in combined)
label_counts: Counter[str] = Counter()
for row in combined:
    label_counts.update(x.strip() for x in row[LABELS].split(",") if x.strip())
print(f"\n-> {OUT}")
print(f"   rows {len(combined):,} | origins {dict(origins)}")
print(f"   distinct labels {len(label_counts)} | labels >= 300 rows: "
      f"{sum(1 for c in label_counts.values() if c >= 300)}")
print(f"   top replication sources: {dict(sources.most_common(6))}")
