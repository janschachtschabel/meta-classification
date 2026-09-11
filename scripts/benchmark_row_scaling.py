"""What does api_v3 cost as the dataset grows - and what breaks first at 600k rows?

Everything measured so far sat at ~26k rows. This measures the actual scaling curve on
real subsets instead of assuming linearity, and extrapolates from the measured slope.

Per row count it reports the four things that decide feasibility:
  - the sparse feature matrix (the dominant memory item)
  - vectorization time
  - one head fit (all labels, threading backend)
  - peak process RSS

Two feature shapes, because they behave very differently: `auto` (word + char n-grams,
the default) and `fast` (word-only). Character n-grams produce far more non-zeros per
document, so this is the real memory lever - unlike the vocabulary caps, which barely
move the matrix (see benchmark_feature_caps.py).

The extrapolation to 600k uses the fitted exponent from log-log regression, so a
super-linear term would show up rather than being assumed away.
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

from app.classifier import make_head  # noqa: E402
from app.data import prepare_targets  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
ROW_COUNTS = [4_000, 8_000, 16_000, 26_000]
TARGET_ROWS = 600_000
SHAPES = {
    # name: (use_char, max_word, max_char)
    "auto (word+char)": (True, 80_000, 120_000),
    "large (word 200k)": (False, 200_000, 0),
    "fast (word 50k)": (False, 50_000, 0),
}
PROC = psutil.Process(os.getpid())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float:
    return PROC.memory_info().rss / 1024**2


def sparse_mb(m) -> float:
    return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1024**2


log("loading data_30k_ai.csv ...")
loaded = load_dataset(
    BASE / "data" / "data_30k_ai.csv",
    ["properties.cclom:title", "properties.cclom:general_description",
     "properties.cclom:general_keyword"],
    "properties.ccm:taxonid", separator=";", label_separator=",",
    label_filter="http://w3id.org/openeduhub/vocabs/discipline/",
)
y_all, classes, row_keep = prepare_targets(loaded.label_lists, min_samples=20)
texts_all = [t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep]
log(f"{len(texts_all)} rows available, {len(classes)} labels, "
    f"label matrix dtype={y_all.dtype} ({y_all.nbytes / 1024**2:.1f} MB)")

rng = np.random.default_rng(SEED)
order = rng.permutation(len(texts_all))

results = []
for shape_name, (use_char, max_word, max_char) in SHAPES.items():
    for n_rows in ROW_COUNTS:
        if n_rows > len(texts_all):
            continue
        idx = order[:n_rows]
        texts = [texts_all[i] for i in idx]
        y = y_all[idx]

        before = rss_mb()
        vec = TfidfBackend(use_char=use_char, max_word_features=max_word,
                           max_char_features=max_char or 1)
        t0 = time.time()
        x = vec.fit_transform(texts)
        vec_s = time.time() - t0
        mat_mb = sparse_mb(x)

        head = make_head(4.0, n_jobs=N_JOBS, solver="newton-cg")
        t0 = time.time()
        with parallel_backend("threading", n_jobs=N_JOBS):
            head.fit(x, y)
        fit_s = time.time() - t0
        peak = rss_mb()

        row = {
            "shape": shape_name, "n_rows": n_rows, "n_features": int(x.shape[1]),
            "nnz": int(x.nnz), "matrix_mb": round(mat_mb, 1),
            "nnz_per_row": round(x.nnz / n_rows, 1),
            "vectorize_seconds": round(vec_s, 1), "fit_seconds": round(fit_s, 1),
            "rss_growth_mb": round(peak - before, 1),
        }
        results.append(row)
        log(f"  {shape_name} {n_rows:>6} rows: matrix {mat_mb:>7.1f} MB "
            f"({row['nnz_per_row']:>5.1f} nnz/row) | vectorize {vec_s:>5.1f}s | "
            f"fit {fit_s:>6.1f}s | RSS +{row['rss_growth_mb']:>6.1f} MB")
        del vec, x, head


def fit_exponent(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Log-log slope + intercept, so a super-linear term shows up instead of being assumed away."""
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(np.exp(intercept))


out = BASE / "row_scaling_results.json"
out.write_text(json.dumps({"target_rows": TARGET_ROWS, "n_labels": len(classes),
                           "label_matrix_dtype": str(y_all.dtype), "results": results},
                          indent=2), encoding="utf-8")
log(f"written: {out}")

print("\n=== SUMMARY (measured) ===")
print(f"{'shape':<18} {'rows':>7} {'features':>9} {'nnz/row':>8} {'matrix MB':>10} "
      f"{'vec s':>7} {'fit s':>8}")
for r in results:
    print(f"{r['shape']:<18} {r['n_rows']:>7} {r['n_features']:>9} {r['nnz_per_row']:>8.1f} "
          f"{r['matrix_mb']:>10.1f} {r['vectorize_seconds']:>7.1f} {r['fit_seconds']:>8.1f}")

print(f"\n=== EXTRAPOLATED to {TARGET_ROWS:,} rows (log-log fit; exponent shown) ===")
print(f"{'shape':<18} {'matrix':>12} {'exp':>6} {'vectorize':>12} {'exp':>6} "
      f"{'one head fit':>14} {'exp':>6}")
for shape_name in SHAPES:
    rows = [r for r in results if r["shape"] == shape_name]
    if len(rows) < 2:
        continue
    xs = [r["n_rows"] for r in rows]
    out_cells = []
    for key, unit in (("matrix_mb", "MB"), ("vectorize_seconds", "s"), ("fit_seconds", "s")):
        slope, coef = fit_exponent(xs, [r[key] for r in rows])
        value = coef * TARGET_ROWS**slope
        text = (f"{value / 1024:.1f} GB" if unit == "MB" and value > 1024 else
                f"{value / 60:.0f} min" if unit == "s" and value > 120 else
                f"{value:.0f} {unit}")
        out_cells.append((text, slope))
    print(f"{shape_name:<18} " + " ".join(f"{t:>12} {s:>6.2f}" for t, s in out_cells))
print("\nExponent ~1.0 = linear in the row count. Above 1.0 would mean it gets worse "
      "than proportionally; below 1.0 means sub-linear (e.g. a saturating vocabulary).")
