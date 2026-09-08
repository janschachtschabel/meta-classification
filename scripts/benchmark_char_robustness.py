"""Do character n-grams buy ROBUSTNESS, not just average F1?

Average F1 said word-only costs -0.0072 - small enough to trade for 10x less memory.
But an average hides the cases character n-grams exist for: a word feature is all-or-
nothing (a typo, an inflection or a compound turns into a different token and the
feature simply vanishes), whereas 3-5 character n-grams still overlap. If that matters,
it shows up as a bigger DROP on hard inputs, not as a lower mean on clean ones.

Both feature shapes are fitted once on the same train split with the same C, and their
thresholds are tuned on the CLEAN validation split - the realistic setup, since nobody
tunes on the noise they have not seen yet. They are then scored on:

  clean            the untouched test split (the reference)
  typos 3% / 8%    character-level noise: swaps, deletions, duplications
  short texts      the shortest quartile (fewest word features to work with)
  OOV-heavy        the quartile with the most tokens missing from the word vocabulary

The number that answers the question is the DROP from each model's own clean score:
whichever degrades less is the more robust representation.
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
SHAPES = {
    "word+char (auto)": {"use_char": True, "max_word_features": 80_000,
                         "max_char_features": 120_000},
    "word only (large)": {"use_char": False, "max_word_features": 200_000,
                          "max_char_features": 1},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_typos(texts: list[str], rate: float, rng: np.random.Generator) -> list[str]:
    """Character-level noise: swap, delete or duplicate, each with probability rate/3."""
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
log(f"{len(texts)} rows, {len(classes)} labels, {len(txt_te)} test rows")

rng = np.random.default_rng(SEED)
CONDITIONS: dict[str, list[str]] = {
    "clean": txt_te,
    "typos 3%": add_typos(txt_te, 0.03, np.random.default_rng(SEED)),
    "typos 8%": add_typos(txt_te, 0.08, np.random.default_rng(SEED + 1)),
}
# Subsets are index-based so the label rows stay aligned.
lengths = np.array([len(t) for t in txt_te])
short_idx = np.argsort(lengths)[: len(txt_te) // 4]

results = []
oov_idx: np.ndarray | None = None
for shape_name, kwargs in SHAPES.items():
    vec = TfidfBackend(**kwargs)
    x_tr = vec.fit_transform(txt_tr)
    head = make_head(C_FIXED, n_jobs=N_JOBS, solver="newton-cg")
    with parallel_backend("threading", n_jobs=N_JOBS):
        head.fit(x_tr, y_tr)

    # Thresholds tuned on the CLEAN validation split, as production does.
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

    # The OOV quartile is defined by the WORD vocabulary, so compute it once and reuse
    # it for both shapes - otherwise the two would be scored on different rows.
    if oov_idx is None:
        vocab = vec.word_vec.vocabulary_
        miss = np.array([
            sum(1 for w in t.lower().split() if w not in vocab) / max(1, len(t.split()))
            for t in txt_te
        ])
        oov_idx = np.argsort(-miss)[: len(txt_te) // 4]
        log(f"OOV-heavy quartile: {miss[oov_idx].mean():.1%} unknown tokens on average "
            f"vs {miss.mean():.1%} overall")

    scores: dict[str, dict] = {}
    for cond, cond_texts in CONDITIONS.items():
        preds = (head.predict_proba(vec.transform(cond_texts)) >= cuts).astype(int)
        scores[cond] = {
            "f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_te, preds, average="micro", zero_division=0)), 4),
        }
    clean_preds = (head.predict_proba(vec.transform(txt_te)) >= cuts).astype(int)
    for cond, idx in (("short texts", short_idx), ("OOV-heavy", oov_idx)):
        scores[cond] = {
            "f1_macro": round(float(f1_score(y_te[idx], clean_preds[idx], average="macro",
                                             zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_te[idx], clean_preds[idx], average="micro",
                                             zero_division=0)), 4),
        }
    results.append({"shape": shape_name, "n_features": int(x_tr.shape[1]), "scores": scores})
    log(f"{shape_name}: " + " | ".join(
        f"{c} {s['f1_micro']:.4f}" for c, s in scores.items()))
    del vec, x_tr, head

out = BASE / "char_robustness_results.json"
out.write_text(json.dumps({"C": C_FIXED, "results": results}, indent=2), encoding="utf-8")
log(f"written: {out}")

CONDS = ["clean", "typos 3%", "typos 8%", "short texts", "OOV-heavy"]
for metric in ("f1_micro", "f1_macro"):
    print(f"\n=== {metric} by condition (drop from each model's OWN clean score) ===")
    print(f"{'shape':<20} " + " ".join(f"{c:>16}" for c in CONDS))
    for r in results:
        clean = r["scores"]["clean"][metric]
        cells = []
        for c in CONDS:
            v = r["scores"][c][metric]
            cells.append(f"{v:.4f}" if c == "clean" else f"{v:.4f} ({v - clean:+.4f})")
        print(f"{r['shape']:<20} " + " ".join(f"{c:>16}" for c in cells))
print("\nRobustness = the smaller drop, not the higher clean score. A word feature is "
      "all-or-nothing under noise; 3-5 char n-grams still overlap.")
