"""Are the TF-IDF vocabulary caps cutting off signal?

Both caps are saturated on data_30k_ai.csv: the deployed model reports exactly
200 000 features, i.e. 80 000 word + 120 000 char, so the vocabulary IS truncated.
Whether the discarded tail carries anything is a different question - this measures it.

The caps are the main RAM lever, so the point is not only "does it help" but "what
does it cost": every variant also reports the vocabulary actually built, the sparse
matrix size and the resulting head size, which is what decides deployability on a
small host.

Identical rows, split, seed, C grid and threshold procedure across variants - the caps
are the only thing that changes. Field weights are left OFF here so this measures the
cap effect alone rather than two changes at once.
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
from app.data import prepare_targets, three_way_split  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
C_GRID = [4.0, 16.0]
# (use_char, word cap, char cap). 80k/120k word+char is the shipped default and the
# saturated baseline. The word-only rows answer a different question: character
# n-grams cost ~12x the non-zeros per document (59 vs 724), which is THE memory lever
# for large datasets — but only if the quality cost is acceptable, which is what these
# measure. The caps alone barely move the matrix.
VARIANTS = [
    ("baseline 80k/120k", True, 80_000, 120_000),
    ("half 40k/60k", True, 40_000, 60_000),
    ("double 160k/240k", True, 160_000, 240_000),
    ("quad 320k/480k", True, 320_000, 480_000),
    ("word-only 50k (fast)", False, 50_000, 1),
    ("word-only 200k", False, 200_000, 1),
]


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
log(f"{len(texts)} rows, {len(classes)} labels")

tr, va, te = three_way_split(len(texts), val_size=0.15, test_size=0.15, seed=SEED)
y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
txt_tr = [texts[i] for i in tr]
txt_va = [texts[i] for i in va]
txt_te = [texts[i] for i in te]


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


results = []
for name, use_char, word_cap, char_cap in VARIANTS:
    vec = TfidfBackend(use_char=use_char, max_word_features=word_cap,
                       max_char_features=char_cap)
    t0 = time.time()
    x_tr = vec.fit_transform(txt_tr)
    vec_s = time.time() - t0
    x_va, x_te = vec.transform(txt_va), vec.transform(txt_te)
    n_word = len(vec.word_vec.vocabulary_)
    n_char = len(vec.char_vec.vocabulary_) if use_char else 0
    saturated = (n_word >= word_cap) and (not use_char or n_char >= char_cap)
    sparse_mb = (x_tr.data.nbytes + x_tr.indices.nbytes + x_tr.indptr.nbytes) / 1024**2
    log(f"{name}: vocab word={n_word} char={n_char} "
        f"{'(BOTH CAPS SATURATED)' if saturated else '(natural size reached)'} "
        f"| matrix {sparse_mb:.0f} MB | vectorize {vec_s:.0f}s")

    best = None
    for c in C_GRID:
        head = make_head(c, n_jobs=N_JOBS, solver="newton-cg")
        t0 = time.time()
        with parallel_backend("threading", n_jobs=N_JOBS):
            head.fit(x_tr, y_tr)
        fit_s = time.time() - t0
        p_va = head.predict_proba(x_va)
        cuts = tune_thresholds(y_va, p_va)
        val = f1_score(y_va, (p_va >= cuts).astype(int), average="macro", zero_division=0)
        log(f"  C={c}: val_f1_macro={val:.4f} (fit {fit_s:.0f}s)")
        if best is None or val > best["val"]:
            best = {"c": c, "val": val, "cuts": cuts, "fit_s": fit_s,
                    "head_mb": sum(e.coef_.nbytes for e in head.estimators_) / 1024**2,
                    "test_proba": head.predict_proba(x_te)}
        del head
    preds = (best["test_proba"] >= best["cuts"]).astype(int)
    row = {
        "variant": name, "use_char": use_char, "word_cap": word_cap, "char_cap": char_cap,
        "vocab_word": n_word, "vocab_char": n_char, "caps_saturated": bool(saturated),
        "nnz_per_row": round(x_tr.nnz / x_tr.shape[0], 1),
        "n_features": int(x_tr.shape[1]), "nnz": int(x_tr.nnz),
        "matrix_mb": round(sparse_mb, 1), "head_mb": round(best["head_mb"], 1),
        "vectorize_seconds": round(vec_s, 1), "fit_seconds": round(best["fit_s"], 1),
        "best_C": best["c"],
        "f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_te, preds, average="micro", zero_division=0)), 4),
    }
    results.append(row)
    log(f"  -> {name}: TEST macro {row['f1_macro']:.4f} micro {row['f1_micro']:.4f}")
    del vec, x_tr, x_va, x_te

out = BASE / "feature_caps_results.json"
out.write_text(json.dumps({"n_rows": len(texts), "results": results}, indent=2), encoding="utf-8")
log(f"written: {out}")

base = next(r for r in results if r["variant"].startswith("baseline"))
print("\n=== SUMMARY (test split; only the vocabulary caps change) ===")
print(f"{'variant':<22} {'vocab w/c':>15} {'nnz/row':>8} {'matrix MB':>10} {'head MB':>8} "
      f"{'f1_macro':>9} {'d_macro':>9} {'f1_micro':>9} {'vec s':>6} {'fit s':>6}")
for r in results:
    print(f"{r['variant']:<22} {str(r['vocab_word']) + '/' + str(r['vocab_char']):>15} "
          f"{r['nnz_per_row']:>8.1f} {r['matrix_mb']:>10.1f} "
          f"{r['head_mb']:>8.1f} {r['f1_macro']:>9.4f} "
          f"{r['f1_macro'] - base['f1_macro']:>+9.4f} {r['f1_micro']:>9.4f} "
          f"{r['vectorize_seconds']:>6.1f} {r['fit_seconds']:>6.1f}")
print("\nnnz/row is what scales the matrix with the row count: at 600k rows a word-only "
      "matrix stays a few hundred MB where word+char reaches several GB.")
