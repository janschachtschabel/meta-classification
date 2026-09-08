"""Does `text_column_weights` actually help? Measured on data_30k_ai.csv.

The feature was built as a lever, not a proven gain. This measures it: identical
rows, identical split/seed, identical C grid and threshold procedure — the ONLY
difference is how the training text is assembled from the three columns.

Caveat this script checks for explicitly: repeating a column makes texts longer, so
a row that failed the `min_text_length` filter unweighted could survive weighted.
If the two configurations do not end up with the same row count, the split indices
no longer refer to the same rows and the comparison is void — the script says so
instead of quietly reporting an invalid delta.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from joblib import parallel_backend  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402

from app.data import load_dataset, prepare_targets, three_way_split  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
C_GRID = [4.0, 16.0, 32.0]
TITLE = "properties.cclom:title"
DESC = "properties.cclom:general_description"
KEYWORD = "properties.cclom:general_keyword"
TEXT_COLS = [TITLE, DESC, KEYWORD]

VARIANTS = {
    "baseline (all 1x)": None,
    "title+keyword x2": {TITLE: 2, KEYWORD: 2},
    "title+keyword x3": {TITLE: 3, KEYWORD: 3},
    # The single-field variants isolate WHICH column carries the gain — the first run
    # showed title-only landing below baseline, so the two are not interchangeable.
    "title x2 only": {TITLE: 2},
    "keyword x2 only": {KEYWORD: 2},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-label cut maximizing F1, searched over that label's own score quantiles."""
    cuts = np.zeros(scores.shape[1])
    qs = np.linspace(0.50, 0.9995, 60)
    for col in range(scores.shape[1]):
        column, truth = scores[:, col], y_true[:, col]
        best_f1, best_t = -1.0, float(np.quantile(column, 0.5))
        for t in np.unique(np.quantile(column, qs)):
            f1 = f1_score(truth, (column >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        cuts[col] = best_t
    return cuts


def metrics(y_true: np.ndarray, scores: np.ndarray, cuts: np.ndarray) -> dict:
    preds = (scores >= cuts).astype(int)
    return {
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
    }


def run(name: str, weights: dict | None) -> dict:
    loaded = load_dataset(
        BASE / "data" / "data_30k_ai.csv", TEXT_COLS, "properties.ccm:taxonid",
        separator=";", label_separator=",",
        label_filter="http://w3id.org/openeduhub/vocabs/discipline/",
        text_column_weights=weights,
    )
    y_all, classes, row_keep = prepare_targets(loaded.label_lists, min_samples=20)
    texts = [t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep]
    log(f"{name}: {len(texts)} rows, {len(classes)} labels")

    train_idx, val_idx, test_idx = three_way_split(
        len(texts), val_size=0.15, test_size=0.15, seed=SEED)
    vec = TfidfBackend()
    x_tr = vec.fit_transform([texts[i] for i in train_idx])
    x_va = vec.transform([texts[i] for i in val_idx])
    x_te = vec.transform([texts[i] for i in test_idx])
    y_tr, y_va, y_te = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    best = None
    for c in C_GRID:
        t0 = time.time()
        head = OneVsRestClassifier(
            LogisticRegression(C=c, class_weight="balanced", max_iter=1000, solver="newton-cg"),
            n_jobs=N_JOBS)
        with parallel_backend("threading", n_jobs=N_JOBS):
            head.fit(x_tr, y_tr)
        fit_s = time.time() - t0
        s_va = head.predict_proba(x_va)
        cuts = tune_thresholds(y_va, s_va)
        val_macro = metrics(y_va, s_va, cuts)["f1_macro"]
        log(f"  {name} C={c}: val_f1_macro={val_macro:.4f} (fit {fit_s:.1f}s)")
        if best is None or val_macro > best["val_macro"]:
            best = {"c": c, "val_macro": val_macro, "cuts": cuts, "fit_s": fit_s,
                    "test_scores": head.predict_proba(x_te)}
    test = metrics(y_te, best["test_scores"], best["cuts"])
    log(f"  -> {name}: best C={best['c']}  TEST f1_macro={test['f1_macro']:.4f} "
        f"f1_micro={test['f1_micro']:.4f}")
    return {"name": name, "weights": weights, "n_rows": len(texts), "n_labels": len(classes),
            "nnz_train": int(x_tr.nnz), "best_C": best["c"],
            "fit_seconds": round(best["fit_s"], 1),
            "val_f1_macro": round(best["val_macro"], 4), "test": test}


results = [run(name, weights) for name, weights in VARIANTS.items()]

row_counts = {r["n_rows"] for r in results}
comparable = len(row_counts) == 1
if not comparable:
    log(f"WARNING: row counts differ across variants {sorted(row_counts)} - the "
        f"min_text_length filter kept different rows, so the splits are NOT the same "
        f"rows and the deltas below are not a clean comparison.")

out = BASE / "field_weights_results.json"
out.write_text(json.dumps({"comparable_row_sets": comparable, "results": results}, indent=2),
               encoding="utf-8")
log(f"written: {out}")

base_macro = results[0]["test"]["f1_macro"]
print("\n=== SUMMARY (test split; only the text assembly differs) ===")
# ASCII only: this runs on a cp1252 Windows console, where a stray Unicode delta
# raises UnicodeEncodeError and throws away a completed 15-minute measurement.
print(f"{'variant':<22} {'rows':>6} {'nnz train':>11} {'best C':>7} "
      f"{'f1_macro':>9} {'d_macro':>9} {'f1_micro':>9} {'fit s':>7}")
for r in results:
    t = r["test"]
    print(f"{r['name']:<22} {r['n_rows']:>6} {r['nnz_train']:>11} {r['best_C']:>7} "
          f"{t['f1_macro']:>9.4f} {t['f1_macro'] - base_macro:>+9.4f} "
          f"{t['f1_micro']:>9.4f} {r['fit_seconds']:>7.1f}")
print(f"\nRow sets identical across variants: {comparable}")
