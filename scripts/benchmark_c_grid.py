"""What does the size of the C grid actually buy - in quality and in time?

The `auto` grid was widened to 8 candidates after a production model selected the old
grid's maximum. That doubled the number of head fits, and under 5-fold CV the fit count
is k x |grid| - which is what makes a large dataset slow. This measures whether the
extra candidates are worth their cost.

Method: fit every candidate ONCE, then evaluate each candidate GRID as a subset of
those fits (a grid's pick is simply its best-scoring member on validation). So the whole
comparison costs 8 fits instead of 14, and the full C curve falls out for free.

Time is then reported as the projected cost of a 5-fold CV run, where the grid size
multiplies the fits but not the per-fold vectorization:
    total = k * (vectorize + |grid| * fit) + deploy(vectorize + fit)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from joblib import parallel_backend  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import load_dataset, prepare_targets, three_way_split  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
K_FOLDS = 5
ALL_C = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
GRIDS = {
    "8 candidates (current auto)": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "4 candidates (4x steps)": [1.0, 4.0, 16.0, 32.0],
    "3 candidates": [2.0, 8.0, 32.0],
    "2 candidates (large)": [4.0, 16.0],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("loading data_30k_ai.csv (same request as faecher_ai_cv5) ...")
loaded = load_dataset(
    BASE / "data" / "data_30k_ai.csv",
    ["properties.cclom:title", "properties.cclom:general_description",
     "properties.cclom:general_keyword"],
    "properties.ccm:taxonid", separator=";", label_separator=",",
    label_filter="http://w3id.org/openeduhub/vocabs/discipline/",
)
y_all, classes, row_keep = prepare_targets(loaded.label_lists, min_samples=20)
texts = [t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep]
tr, va, te = three_way_split(len(texts), val_size=0.15, test_size=0.15, seed=SEED)
y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
log(f"{len(texts)} rows, {len(classes)} labels")

vec = TfidfBackend()
t0 = time.time()
x_tr = vec.fit_transform([texts[i] for i in tr])
vectorize_s = time.time() - t0
x_va, x_te = vec.transform([texts[i] for i in va]), vec.transform([texts[i] for i in te])
log(f"vectorized in {vectorize_s:.1f}s ({x_tr.shape[1]} features)")


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    cuts = np.zeros(proba.shape[1])
    qs = np.linspace(0.50, 0.9995, 60)
    for col in range(proba.shape[1]):
        column, truth = proba[:, col], y_true[:, col]
        best_f1, best_t = -1.0, float(np.quantile(column, 0.5))
        for t in np.unique(np.quantile(column, qs)):
            f1 = f1_score(truth, (column >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        cuts[col] = best_t
    return cuts


# One fit per candidate; every grid is then a subset of these results.
curve = {}
fit_times = []
for c in ALL_C:
    head = make_head(c, n_jobs=N_JOBS, solver="newton-cg")
    t0 = time.time()
    with parallel_backend("threading", n_jobs=N_JOBS):
        head.fit(x_tr, y_tr)
    fit_s = time.time() - t0
    fit_times.append(fit_s)
    p_va = head.predict_proba(x_va)
    cuts = tune_thresholds(y_va, p_va)
    val = float(f1_score(y_va, (p_va >= cuts).astype(int), average="macro", zero_division=0))
    p_te = head.predict_proba(x_te)
    preds = (p_te >= cuts).astype(int)
    curve[c] = {
        "val_f1_macro": round(val, 4), "fit_seconds": round(fit_s, 1),
        "test_f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
        "test_f1_micro": round(float(f1_score(y_te, preds, average="micro", zero_division=0)), 4),
    }
    log(f"  C={c:<5}: val {val:.4f} | test macro {curve[c]['test_f1_macro']:.4f} "
        f"micro {curve[c]['test_f1_micro']:.4f} (fit {fit_s:.1f}s)")
    del head

mean_fit = sum(fit_times) / len(fit_times)
results = []
for name, grid in GRIDS.items():
    picked = max(grid, key=lambda c: curve[c]["val_f1_macro"])
    # A grid's cost under k-fold CV: the fits multiply, the vectorization does not.
    cv_seconds = K_FOLDS * (vectorize_s + len(grid) * mean_fit) + vectorize_s + mean_fit
    results.append({
        "grid": name, "candidates": grid, "picked_C": picked,
        "val_f1_macro": curve[picked]["val_f1_macro"],
        "test_f1_macro": curve[picked]["test_f1_macro"],
        "test_f1_micro": curve[picked]["test_f1_micro"],
        "cv5_minutes_at_this_size": round(cv_seconds / 60, 1),
    })

out = BASE / "c_grid_results.json"
out.write_text(json.dumps({"n_rows": len(texts), "vectorize_seconds": round(vectorize_s, 1),
                           "mean_fit_seconds": round(mean_fit, 1), "curve": curve,
                           "results": results}, indent=2), encoding="utf-8")
log(f"written: {out}")

print("\n=== C curve (one fit per candidate, identical split and thresholds) ===")
print(f"{'C':>6} {'val macro':>10} {'test macro':>11} {'test micro':>11} {'fit s':>7}")
for c in ALL_C:
    e = curve[c]
    print(f"{c:>6} {e['val_f1_macro']:>10.4f} {e['test_f1_macro']:>11.4f} "
          f"{e['test_f1_micro']:>11.4f} {e['fit_seconds']:>7.1f}")

base = results[0]
print("\n=== What each grid picks, and what it costs ===")
print(f"{'grid':<28} {'picks C':>8} {'test macro':>11} {'d_macro':>9} {'test micro':>11} "
      f"{'CV5 min':>9} {'saved':>8}")
for r in results:
    print(f"{r['grid']:<28} {r['picked_C']:>8} {r['test_f1_macro']:>11.4f} "
          f"{r['test_f1_macro'] - base['test_f1_macro']:>+9.4f} {r['test_f1_micro']:>11.4f} "
          f"{r['cv5_minutes_at_this_size']:>9.1f} "
          f"{base['cv5_minutes_at_this_size'] - r['cv5_minutes_at_this_size']:>7.1f}m")
print(f"\nCV5 minutes are for THIS dataset ({len(texts)} rows); the fit share scales with "
      "the row count, the vectorization share stays one pass per fold.")
