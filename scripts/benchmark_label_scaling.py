"""How does the deployed head behave as a vocab grows from 48 to ~300 labels?

Motivated by WLO vocabs like `ccm:curriculum`, which carries hundreds of labels
rather than the 48 of the subject vocab. Measures COST and the DECISION BEHAVIOUR,
not tuned quality: one fixed C, so the F1 columns are context, not a verdict.

Design: the label set is the only thing that changes. Same texts, same TF-IDF matrix
(fit once), same split - restricted to the N most frequent labels for N in {48, 300}.
So the 48 -> 300 ratio is measured on identical data instead of extrapolated.

The headline result is the precision/recall pair together with the "labels asserted
per row" column: per-label thresholds are tuned to maximize each label's OWN F1, and
on rare labels that buys recall with false positives. Macro dilutes those 1/N, micro
pools them - which is why micro collapses while macro only sags.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from joblib import parallel_backend  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import load_dataset, prepare_targets, three_way_split  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
C_FIXED = 4.0
LABEL_COUNTS = [48, 300]
PROC = psutil.Process(os.getpid())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float:
    return PROC.memory_info().rss / 1024**2


log("loading data_30k_ai.csv with ccm:curriculum as the label column ...")
loaded = load_dataset(
    BASE / "data" / "data_30k_ai.csv",
    ["properties.cclom:title", "properties.cclom:general_description",
     "properties.cclom:general_keyword"],
    "properties.ccm:curriculum",
    separator=";", label_separator=",",
)
y_full, classes_full, row_keep = prepare_targets(loaded.label_lists, min_samples=20)
texts = [t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep]
log(f"{len(texts)} rows, {len(classes_full)} labels available (min_samples=20)")

# Keep the N most frequent labels, so a smaller N is a strict subset of a larger one.
order = np.argsort(-y_full.sum(axis=0))
train_idx, val_idx, test_idx = three_way_split(
    len(texts), val_size=0.15, test_size=0.15, seed=SEED)

log("fitting TF-IDF once (shared by every configuration) ...")
vec = TfidfBackend()
x_tr = vec.fit_transform([texts[i] for i in train_idx])
x_va = vec.transform([texts[i] for i in val_idx])
x_te = vec.transform([texts[i] for i in test_idx])
log(f"features: {x_tr.shape[1]}, nnz={x_tr.nnz}, RSS after features: {rss_mb():.0f} MB")


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-label cut maximizing that label's own F1 - the production rule."""
    cuts = np.zeros(proba.shape[1])
    qs = np.linspace(0.50, 0.9995, 40)
    for col in range(proba.shape[1]):
        column, truth = proba[:, col], y_true[:, col]
        best_f1, best_t = -1.0, float(np.quantile(column, 0.5))
        for t in np.unique(np.quantile(column, qs)):
            f1 = f1_score(truth, (column >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        cuts[col] = best_t
    return cuts


results = []
for n_labels in LABEL_COUNTS:
    cols = np.sort(order[:n_labels])
    y_tr = y_full[np.ix_(train_idx, cols)]
    y_va = y_full[np.ix_(val_idx, cols)]
    y_te = y_full[np.ix_(test_idx, cols)]
    log(f"--- {n_labels} labels (rarest kept label has "
        f"{int(y_full[:, cols].sum(axis=0).min())} rows) ---")

    before = rss_mb()
    head = make_head(C_FIXED, n_jobs=N_JOBS, solver="newton-cg")
    t0 = time.time()
    with parallel_backend("threading", n_jobs=N_JOBS):
        head.fit(x_tr, y_tr)
    fit_s = time.time() - t0
    peak = rss_mb()
    coef_bytes = sum(e.coef_.nbytes for e in head.estimators_)

    t1 = time.time()
    cuts = tune_thresholds(y_va, head.predict_proba(x_va))
    tune_s = time.time() - t1
    preds = (head.predict_proba(x_te) >= cuts).astype(int)
    row = {
        "n_labels": n_labels, "fit_seconds": round(fit_s, 1),
        "coef_dtype": str(head.estimators_[0].coef_.dtype),
        "coef_mb": round(coef_bytes / 1024**2, 1),
        "rss_growth_mb": round(peak - before, 1),
        "threshold_tuning_seconds": round(tune_s, 1),
        "f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_te, preds, average="micro", zero_division=0)), 4),
        "precision_micro": round(
            float(precision_score(y_te, preds, average="micro", zero_division=0)), 4),
        "recall_micro": round(
            float(recall_score(y_te, preds, average="micro", zero_division=0)), 4),
        "predicted_labels_per_row": round(float(preds.sum(axis=1).mean()), 2),
        "true_labels_per_row": round(float(y_te.sum(axis=1).mean()), 2),
    }
    results.append(row)
    log(f"  fit {fit_s:>6.1f}s | coef {row['coef_mb']:>6.1f} MB {row['coef_dtype']} | "
        f"RSS +{row['rss_growth_mb']:>6.1f} MB | thresholds {tune_s:>5.1f}s | "
        f"macro {row['f1_macro']:.4f} micro {row['f1_micro']:.4f} "
        f"(P {row['precision_micro']:.3f} R {row['recall_micro']:.3f}) | "
        f"labels/row predicted {row['predicted_labels_per_row']} "
        f"vs true {row['true_labels_per_row']}")
    del head

out = BASE / "label_scaling_results.json"
out.write_text(json.dumps({"label_column": "properties.ccm:curriculum", "head": "LogisticRegression",
                           "solver": "newton-cg", "C": C_FIXED, "n_rows": len(texts),
                           "results": results}, indent=2), encoding="utf-8")
log(f"written: {out}")

# ASCII only: this runs on a cp1252 console, where a stray Unicode character raises
# UnicodeEncodeError and throws away a completed multi-minute measurement.
print("\n=== SUMMARY (same texts/features/split; only the label count changes) ===")
print(f"{'labels':>7} {'fit s':>8} {'coef MB':>9} {'RSS +MB':>9} {'thresh s':>9} "
      f"{'f1_macro':>9} {'f1_micro':>9} {'P_micro':>8} {'R_micro':>8} {'pred/row':>9} {'true/row':>9}")
for r in results:
    print(f"{r['n_labels']:>7} {r['fit_seconds']:>8.1f} {r['coef_mb']:>9.1f} "
          f"{r['rss_growth_mb']:>9.1f} {r['threshold_tuning_seconds']:>9.1f} "
          f"{r['f1_macro']:>9.4f} {r['f1_micro']:>9.4f} {r['precision_micro']:>8.4f} "
          f"{r['recall_micro']:>8.4f} {r['predicted_labels_per_row']:>9.2f} "
          f"{r['true_labels_per_row']:>9.2f}")

a, b = results[0], results[-1]
factor = b["n_labels"] / a["n_labels"]
print(f"\n--- scaling {a['n_labels']} -> {b['n_labels']} labels (factor {factor:.2f}x) ---")
print(f"fit x{b['fit_seconds'] / a['fit_seconds']:.2f}  "
      f"coef x{b['coef_mb'] / a['coef_mb']:.2f}  "
      f"thresholds x{b['threshold_tuning_seconds'] / a['threshold_tuning_seconds']:.2f}  "
      f"labels asserted per row {a['predicted_labels_per_row']} -> {b['predicted_labels_per_row']} "
      f"(true {a['true_labels_per_row']} -> {b['true_labels_per_row']})")
