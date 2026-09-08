"""Is there a middle ground between "3.2 GB and robust" and "479 MB and fragile"?

Measured so far: character n-grams cost ~10x the memory but degrade 3.6x less under
text noise. Both extremes are unattractive for a large corpus of real editorial data.

The cost driver is non-zeros per document, and for character n-grams that is set by the
n-gram RANGE, not by the vocabulary cap (halving the caps moved nnz/row by 6%). The
default `char_wb (3, 5)` emits three n-grams per character position; (4, 5) emits two,
(5, 5) one. So narrowing the range should cut the matrix roughly proportionally - the
question is how much robustness goes with it.

Each variant is fitted once at a fixed C on the same split, thresholds tuned on the
clean validation split, then scored clean and under 8% character noise. The columns that
decide it are nnz/row (what it costs at 600k rows) and the noise drop (what it buys).
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
C_FIXED = 16.0
NOISE = 0.08
TARGET_ROWS = 600_000

# name -> TfidfBackend kwargs
VARIANTS = {
    "word+char 3-5 (auto)": {"use_char": True, "char_ngram": (3, 5),
                             "max_word_features": 80_000, "max_char_features": 120_000},
    "word+char 3-4": {"use_char": True, "char_ngram": (3, 4),
                      "max_word_features": 80_000, "max_char_features": 120_000},
    "word+char 4-5": {"use_char": True, "char_ngram": (4, 5),
                      "max_word_features": 80_000, "max_char_features": 120_000},
    "word+char 5-5": {"use_char": True, "char_ngram": (5, 5),
                      "max_word_features": 80_000, "max_char_features": 120_000},
    "word only": {"use_char": False, "max_word_features": 200_000, "max_char_features": 1},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score(head, vec, cuts, texts_in: list[str], y_true) -> tuple[float, float]:
    """Macro/micro F1 of one fitted (vectorizer, head, thresholds) on given texts.

    Takes its dependencies explicitly rather than closing over the loop variables:
    each iteration releases them, which would leave a closure pointing at nothing.
    """
    preds = (head.predict_proba(vec.transform(texts_in)) >= cuts).astype(int)
    return (round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
            round(float(f1_score(y_true, preds, average="micro", zero_division=0)), 4))


def add_typos(texts: list[str], rate: float, rng: np.random.Generator) -> list[str]:
    out = []
    for text in texts:
        chars = list(text)
        i = 0
        while i < len(chars):
            if rng.random() < rate:
                op = rng.integers(0, 3)
                if op == 0 and i + 1 < len(chars):
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 1
                elif op == 1:
                    del chars[i]
                    continue
                else:
                    chars.insert(i, chars[i])
                    i += 1
            i += 1
        out.append("".join(chars))
    return out


log("loading data_30k_ai.csv ...")
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
txt_tr = [texts[i] for i in tr]
txt_va = [texts[i] for i in va]
txt_te = [texts[i] for i in te]
txt_noisy = add_typos(txt_te, NOISE, np.random.default_rng(SEED))
log(f"{len(texts)} rows, {len(classes)} labels")

results = []
for name, kwargs in VARIANTS.items():
    vec = TfidfBackend(**kwargs)
    t0 = time.time()
    x_tr = vec.fit_transform(txt_tr)
    vec_s = time.time() - t0
    nnz_per_row = x_tr.nnz / x_tr.shape[0]

    head = make_head(C_FIXED, n_jobs=N_JOBS, solver="newton-cg")
    t0 = time.time()
    with parallel_backend("threading", n_jobs=N_JOBS):
        head.fit(x_tr, y_tr)
    fit_s = time.time() - t0

    p_va = head.predict_proba(vec.transform(txt_va))
    cuts = np.zeros(p_va.shape[1])
    for col in range(p_va.shape[1]):
        column, truth = p_va[:, col], y_va[:, col]
        best_f1, best_t = -1.0, float(np.quantile(column, 0.5))
        for t in np.unique(np.quantile(column, np.linspace(0.50, 0.9995, 60))):
            f1 = f1_score(truth, (column >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        cuts[col] = best_t

    clean_macro, clean_micro = score(head, vec, cuts, txt_te, y_te)
    noisy_macro, noisy_micro = score(head, vec, cuts, txt_noisy, y_te)
    # 8 bytes per non-zero (float32 value + int32 index) is the CSR cost.
    projected_gb = nnz_per_row * TARGET_ROWS * 8 / 1024**3
    row = {
        "variant": name, "n_features": int(x_tr.shape[1]),
        "nnz_per_row": round(nnz_per_row, 1),
        "projected_matrix_gb_at_600k": round(projected_gb, 2),
        "vectorize_seconds": round(vec_s, 1), "fit_seconds": round(fit_s, 1),
        "clean_macro": clean_macro, "clean_micro": clean_micro,
        "noisy_macro": noisy_macro, "noisy_micro": noisy_micro,
        "noise_drop_micro": round(noisy_micro - clean_micro, 4),
        "noise_drop_macro": round(noisy_macro - clean_macro, 4),
    }
    results.append(row)
    log(f"  {name:<22} nnz/row {nnz_per_row:>6.1f} | clean {clean_micro:.4f} | "
        f"noisy {noisy_micro:.4f} ({row['noise_drop_micro']:+.4f}) | "
        f"@600k ~{projected_gb:.2f} GB | fit {fit_s:.1f}s")
    del vec, x_tr, head

out = BASE / "char_range_results.json"
out.write_text(json.dumps({"C": C_FIXED, "noise": NOISE, "results": results}, indent=2),
               encoding="utf-8")
log(f"written: {out}")

print(f"\n=== character n-gram range: cost vs. noise robustness (C={C_FIXED}, "
      f"{int(NOISE * 100)}% char noise) ===")
print(f"{'variant':<22} {'nnz/row':>8} {'@600k GB':>9} {'clean mic':>10} {'noisy mic':>10} "
      f"{'drop mic':>9} {'clean mac':>10} {'drop mac':>9} {'fit s':>7}")
for r in results:
    print(f"{r['variant']:<22} {r['nnz_per_row']:>8.1f} "
          f"{r['projected_matrix_gb_at_600k']:>9.2f} {r['clean_micro']:>10.4f} "
          f"{r['noisy_micro']:>10.4f} {r['noise_drop_micro']:>+9.4f} "
          f"{r['clean_macro']:>10.4f} {r['noise_drop_macro']:>+9.4f} {r['fit_seconds']:>7.1f}")
print("\nPick by: how much matrix can you afford at 600k rows, and how much noise drop "
      "is acceptable. Word-only is the cheapest and the most fragile.")
