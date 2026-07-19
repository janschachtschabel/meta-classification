"""Enrich weak (tail) subjects of a curated dataset with real rows from a large
uncurated dataset — a standalone data tool, deliberately NOT part of the API.

Discipline (the one rule that keeps the experiment honest): a holdout slice of
the CURATED data is split off FIRST and never mixed with anything; enriched
rows go into training only. Mined rows are marked in a `source` column.

Usage (from api_v3/):
    python scripts/enrich_tail.py --curated data/data_30k.csv \
        --raw ../data/data_300k.csv --metrics models/faecher_cv5/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data import clean_text, split_labels  # noqa: E402

TEXT_COLS = ["properties.cclom:title", "properties.cclom:general_description",
             "properties.cclom:general_keyword"]
LABEL_COL = "properties.ccm:taxonid"
FILT = "http://w3id.org/openeduhub/vocabs/discipline/"


def combined_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    combined = df[text_cols[0]].fillna("")
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna("")
    return combined.map(clean_text)


def select_weak(metrics: dict, max_f1: float) -> list[str]:
    """Subjects whose per-label F1 is below the threshold (sorted, worst first)."""
    plf = metrics["metrics"]["per_label_f1"]
    return [u for u, f1 in sorted(plf.items(), key=lambda x: x[1]) if f1 < max_f1]


def _labels(cell: object, filt: str) -> list[str]:
    return [x for x in split_labels(cell, ",") if filt in x]


def split_holdout(df: pd.DataFrame, text_cols: list[str], label_col: str, filt: str,
                  *, weak: set[str], seed: int,
                  p_weak: float = 0.4, p_rest: float = 0.15):
    """Deduplicate by cleaned text (mirrors the training pipeline), then split a
    seeded holdout: rows carrying a weak subject get a higher holdout share so
    the per-subject measurement has real mass."""
    df = df.copy()
    df["__text"] = combined_text(df, text_cols)
    df = df[df["__text"].str.len() >= 5].drop_duplicates("__text", keep="first")
    rng = random.Random(seed)
    is_hold = []
    for cell in df[label_col]:
        has_weak = any(lab in weak for lab in _labels(cell, filt))
        is_hold.append(rng.random() < (p_weak if has_weak else p_rest))
    mask = pd.Series(is_hold, index=df.index)
    return (df[~mask].drop(columns="__text"), df[mask].drop(columns="__text"))


def mine_rows(raw: pd.DataFrame, text_cols: list[str], label_col: str, filt: str,
              *, weak: set[str], needed: dict[str, int], known_texts: set[str],
              min_text_len: int = 30) -> pd.DataFrame:
    """Greedily take raw rows that serve a still-needed weak subject: long
    enough text, unseen text (no dupes vs curated data or within the mined set)."""
    remaining = dict(needed)
    picked: list[int] = []
    texts = combined_text(raw, text_cols)
    for idx, (text, cell) in enumerate(zip(texts, raw[label_col], strict=False)):
        if all(v <= 0 for v in remaining.values()):
            break
        if len(text) < min_text_len or text in known_texts:
            continue
        hits = [lab for lab in _labels(cell, filt) if remaining.get(lab, 0) > 0]
        if not hits:
            continue
        known_texts.add(text)
        picked.append(idx)
        for lab in hits:
            remaining[lab] -= 1
    return raw.iloc[picked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--metrics", required=True, help="metrics.json of the reference model")
    ap.add_argument("--max-f1", type=float, default=0.45)
    ap.add_argument("--target", type=int, default=300, help="target support per weak subject")
    ap.add_argument("--min-text-len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    weak = select_weak(metrics, args.max_f1)
    print(f"Weak subjects (F1 < {args.max_f1}): {len(weak)}")

    curated = pd.read_csv(args.curated, sep=";", dtype=str, encoding="utf-8", low_memory=False)
    train, holdout = split_holdout(curated, TEXT_COLS, LABEL_COL, FILT,
                                   weak=set(weak), seed=args.seed)

    def support(df: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in df[LABEL_COL]:
            for lab in _labels(cell, FILT):
                counts[lab] = counts.get(lab, 0) + 1
        return counts

    train_sup, hold_sup = support(train), support(holdout)
    needed = {u: max(0, args.target - train_sup.get(u, 0)) for u in weak}

    known = set(combined_text(train, TEXT_COLS)) | set(combined_text(holdout, TEXT_COLS))
    mined_parts = []
    for chunk in pd.read_csv(args.raw, sep=";", dtype=str, encoding="utf-8",
                             usecols=lambda c: c in set(curated.columns), chunksize=50_000):
        mined_parts.append(mine_rows(chunk, TEXT_COLS, LABEL_COL, FILT, weak=set(weak),
                                     needed=needed, known_texts=known,
                                     min_text_len=args.min_text_len))
        got = support(pd.concat(mined_parts)) if mined_parts else {}
        needed = {u: max(0, args.target - train_sup.get(u, 0) - got.get(u, 0)) for u in weak}
        if all(v <= 0 for v in needed.values()):
            break
    mined = pd.concat(mined_parts) if mined_parts else pd.DataFrame(columns=curated.columns)
    mined = mined.reindex(columns=curated.columns, fill_value="")

    train, holdout, mined = train.copy(), holdout.copy(), mined.copy()
    train["source"], holdout["source"], mined["source"] = "curated", "curated", "mined"
    enriched = pd.concat([train, mined], ignore_index=True)

    out = Path(args.outdir)
    enriched.to_csv(out / "data_30k_enriched.csv", sep=";", index=False, encoding="utf-8")
    train.to_csv(out / "data_30k_base.csv", sep=";", index=False, encoding="utf-8")
    holdout.to_csv(out / "holdout_curated.csv", sep=";", index=False, encoding="utf-8")

    mined_sup = support(mined)
    print(f"\ncurated train rows: {len(train)} | holdout rows: {len(holdout)} "
          f"| mined rows added: {len(mined)}")
    print(f"{'subject':<52}{'train':>7}{'+mined':>8}{'holdout':>9}")
    for u in weak:
        print(f"{u.rsplit('/', 1)[-1]:<52}{train_sup.get(u, 0):>7}"
              f"{mined_sup.get(u, 0):>8}{hold_sup.get(u, 0):>9}")


if __name__ == "__main__":
    main()
