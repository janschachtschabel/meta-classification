"""Compare models on the curated holdout produced by enrich_tail.py — the only
honest yardstick for the enrichment experiment (holdout texts were never part
of any training data, curated labels only).

Usage (from api_v3/, server running):
    python scripts/eval_holdout.py --models exp_base,exp_enriched \
        --holdout data/holdout_curated.csv --key ui-ro-key
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from enrich_tail import FILT, LABEL_COL, select_weak  # noqa: E402  (sibling script)

from app.data import split_labels  # noqa: E402
from app.dataset_load import combine_text_columns  # noqa: E402


def api(base: str, key: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def mean(values) -> float | None:
    """None for nothing to average: with no shared or no weak label the macro averages
    divided by zero, after the whole holdout had been classified (audit 2026-09-30, W05)."""
    values = list(values)
    return sum(values) / len(values) if values else None


def model_texts(df: pd.DataFrame, info: dict) -> list[str]:
    """The holdout text as THIS model was trained on it: its own text columns, each repeated
    as its weights say -- the app's own assembly. Every column once judged a model trained
    with the title twice on a distribution it never saw, its thresholds cutting in the wrong
    place (audit 2026-09-30, W05)."""
    metadata = info.get("metadata") or {}
    columns = list(metadata.get("text_columns") or [])
    missing = [c for c in columns if c not in df.columns]
    if not columns or missing:
        raise SystemExit(f"the holdout lacks the model's text columns {missing or '(none recorded)'}")
    weights = {c: w for c, w in (metadata.get("text_column_weights") or {}).items() if c in columns}
    return combine_text_columns(df, columns, weights).tolist()


def evaluate(base: str, key: str, model: str, texts: list[str],
             y_true: list[set[str]], label_space: set[str]) -> dict:
    pred: list[set[str]] = []
    for start in range(0, len(texts), 500):
        r = api(base, key, "/predict/batch",
                {"texts": texts[start:start + 500], "model_name": model})
        pred += [{p["uri"] for p in row["predictions"]} for row in r["results"]]

    per_label: dict[str, tuple[int, int, int]] = {}
    micro_tp = micro_fp = micro_fn = 0
    for uri in label_space:
        tp = sum(1 for t, p in zip(y_true, pred, strict=False) if uri in t and uri in p)
        fp = sum(1 for t, p in zip(y_true, pred, strict=False) if uri not in t and uri in p)
        fn = sum(1 for t, p in zip(y_true, pred, strict=False) if uri in t and uri not in p)
        per_label[uri] = (tp, fp, fn)
        micro_tp, micro_fp, micro_fn = micro_tp + tp, micro_fp + fp, micro_fn + fn
    scores = {u: f1(*c) for u, c in per_label.items()}
    return {
        "micro": f1(micro_tp, micro_fp, micro_fn),
        "macro": mean(scores.values()),
        "per_label": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--holdout", default="data/holdout_curated.csv")
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    ap.add_argument("--key", default="ui-ro-key")
    ap.add_argument("--weak-metrics", default="models/faecher_cv5/metrics.json")
    ap.add_argument("--max-f1", type=float, default=0.45)
    args = ap.parse_args()
    models = args.models.split(",")

    df = pd.read_csv(args.holdout, sep=";", dtype=str, encoding="utf-8", low_memory=False)
    y_true = [{x for x in split_labels(c, ",") if FILT in x} for c in df[LABEL_COL]]
    infos = {m: api(args.base, args.key, f"/models/{m}") for m in models}
    texts = {m: model_texts(df, infos[m]) for m in models}

    # Fair comparison: only labels every model can predict AND the holdout contains.
    class_sets = [set(infos[m]["classes"]) for m in models]
    holdout_labels = set().union(*y_true)
    label_space = set.intersection(*class_sets) & holdout_labels
    if not label_space:
        raise SystemExit("the models and the holdout share no label: nothing to compare")
    weak = [u for u in select_weak(json.loads(Path(args.weak_metrics).read_text(encoding="utf-8")),
                                   args.max_f1) if u in label_space]
    print(f"holdout rows: {len(df)} | shared label space: {len(label_space)} "
          f"| weak subjects evaluated: {len(weak)}")

    results = {m: evaluate(args.base, args.key, m, texts[m], y_true, label_space) for m in models}
    header = "".join(f"{m:>16}" for m in models)
    print(f"\n{'':<40}{header}")
    for metric in ("micro", "macro"):
        print(f"{'F1 ' + metric:<40}" + "".join(f"{results[m][metric]:>16.3f}" for m in models))
    if not weak:
        print("\nno weak subject in the shared label space")
        return
    print(f"\n{'weak subject':<40}{header}")
    for u in weak:
        print(f"{u.rsplit('/', 1)[-1]:<40}"
              + "".join(f"{results[m]['per_label'][u]:>16.3f}" for m in models))
    weak_macros = {m: mean(results[m]["per_label"][u] for u in weak) for m in models}
    print(f"{'MACRO over weak subjects':<40}" + "".join(f"{weak_macros[m]:>16.3f}" for m in models))


if __name__ == "__main__":
    main()
