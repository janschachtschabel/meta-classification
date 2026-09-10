"""Does multilabel-aware splitting make the numbers steadier?

`Profile.stratified_splits` replaces the random fold/holdout split with iterative
stratification, so every label keeps its share in every subset instead of only the row
count being balanced. Nothing here starves to zero — `prepare_targets` floors each label
at `min_samples` — so the claim under test is not "rare labels become learnable" but
"the evaluation stops wobbling because of which rows happened to land where".

**Gate for making it a default (plan item B3): mean macro F1 not worse, AND per-label F1
varying less across seeds.** The second half is the point; the first is the guard that
it was not bought by making the model worse.

**The design isolates the splitter on purpose.** `C` is fixed, so the deployed model is
byte-identical for every seed and both arms — the ONLY thing that moves is the set of
thresholds, which is read off the out-of-fold probabilities and therefore depends
entirely on the fold draw. The held-out test rows are also fixed across every run. Left
free, the C search and a re-drawn test set would both contribute variance of their own
and drown the effect being measured.

One more trap this had to avoid: if the test split itself were stratified in the
stratified arm, part of any variance reduction would come from scoring on a steadier
test set rather than from a steadier model. Both arms score the identical rows.

Usage:
    python scripts/benchmark_stratified_splits.py [--seeds 5] [--c 32] [--rows N]
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
from joblib import parallel_backend  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import load_dataset, prepare_targets, three_way_split  # noqa: E402
from app.profiles import load_training_config  # noqa: E402
from app.settings import get_settings  # noqa: E402
from app.thresholds import apply_thresholds  # noqa: E402
from app.tuning import cross_val_evaluate  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

N_JOBS = 6
K_FOLDS = 3
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"
TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"

MAX_F1_LOSS = 0.002

_SHIPPED = load_training_config(get_settings().config_file).get("auto")
SELECT_ON_TUNED = _SHIPPED.select_c_on_tuned_thresholds


def log(msg: str) -> None:
    print(msg, flush=True)


def threshold_run(matrix, y_pool, classes, *, seed: int, stratified: bool, c: float):
    """One cross-validation over the pool; returns the thresholds it produces."""
    with parallel_backend("threading", n_jobs=N_JOBS):
        _best_c, global_t, per_label_t, _metrics = cross_val_evaluate(
            TfidfBackend, [""] * y_pool.shape[0], y_pool, classes, matrix=matrix,
            k=K_FOLDS, c_grid=[c], seed=seed, n_jobs=N_JOBS, stratified=stratified,
            # A one-candidate grid makes this a no-op today — both selection rules
            # tune the same thresholds off the same out-of-fold probabilities, checked
            # directly — but it is read from the shipped profile anyway, so widening
            # the grid later cannot silently measure against a rule nobody ships.
            select_on_tuned_thresholds=SELECT_ON_TUNED,
        )
    return global_t, per_label_t


def summarise(per_label_f1: np.ndarray, macro: np.ndarray) -> dict:
    """per_label_f1 is (seeds x labels); the spread across seeds is the whole question."""
    return {
        "macro_f1_mean": round(float(macro.mean()), 6),
        "macro_f1_std": round(float(macro.std()), 6),
        "per_label_f1_std_mean": round(float(per_label_f1.std(axis=0).mean()), 6),
        "per_label_f1_std_worst": round(float(per_label_f1.std(axis=0).max()), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5,
                        help="how many fold draws to average the spread over")
    parser.add_argument("--c", type=float, default=32.0,
                        help="fixed regularization: holding it still is what isolates "
                             "the splitter from the C search")
    args = parser.parse_args()

    data = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                        label_filter=DISCIPLINE)
    texts, label_lists = data.texts, data.label_lists
    if args.rows:
        texts, label_lists = texts[: args.rows], label_lists[: args.rows]
    y, classes, keep = prepare_targets(label_lists, min_samples=20)
    texts = [text for text, keeper in zip(texts, keep, strict=False) if keeper]

    # Fixed for every run, and drawn the SAME way for both arms.
    train_idx, val_idx, test_idx = three_way_split(
        len(texts), val_size=0.2, test_size=0.2, seed=0)
    pool = np.concatenate([train_idx, val_idx])
    y_pool, y_test = y[pool], y[test_idx]
    support = y_pool.sum(axis=0)
    log(f"school subjects (discipline): {len(texts)} rows x {len(classes)} labels, "
        f"{len(pool)} pooled, {len(test_idx)} held out (fixed), C={args.c:g}, "
        f"{args.seeds} fold draws")
    log(f"  label support in the pool: min {int(support.min())}, "
        f"median {int(np.median(support))}, max {int(support.max())}")

    vectorizer = TfidfBackend()
    x_pool = vectorizer.fit_transform([texts[i] for i in pool])
    with parallel_backend("threading", n_jobs=N_JOBS):
        head = make_head(args.c, n_jobs=N_JOBS, solver="newton-cg")
        head.fit(x_pool, y_pool)
        proba_test = head.predict_proba(vectorizer.transform([texts[i] for i in test_idx]))
    log("  deploy model fitted once — identical for every seed and both arms")

    results = {}
    for arm, stratified in (("random folds (today)", False), ("stratified folds", True)):
        per_label, macro = [], []
        for seed in range(1, args.seeds + 1):
            global_t, per_label_t = threshold_run(
                x_pool, y_pool, classes, seed=seed, stratified=stratified, c=args.c)
            preds = apply_thresholds(proba_test, classes, global_t, per_label_t)
            per_label.append(f1_score(y_test, preds, average=None, zero_division=0))
            macro.append(f1_score(y_test, preds, average="macro", zero_division=0))
        results[arm] = summarise(np.array(per_label), np.array(macro))
        s = results[arm]
        log(f"  {arm:22}: macro {s['macro_f1_mean']:.4f} (sd {s['macro_f1_std']:.4f})   "
            f"per-label F1 sd across seeds: mean {s['per_label_f1_std_mean']:.4f}, "
            f"worst {s['per_label_f1_std_worst']:.4f}")

    base, strat = results["random folds (today)"], results["stratified folds"]
    f1_delta = strat["macro_f1_mean"] - base["macro_f1_mean"]
    spread_delta = strat["per_label_f1_std_mean"] - base["per_label_f1_std_mean"]
    passes = f1_delta >= -MAX_F1_LOSS and spread_delta < 0
    report = {
        "rows": len(texts), "labels": len(classes), "seeds": args.seeds, "C": args.c,
        "pool_rows": len(pool), "held_out_rows": len(test_idx),
        "results": results,
        "delta_macro_f1": round(f1_delta, 6),
        "delta_per_label_f1_std": round(spread_delta, 6),
        "passes_gate": passes,
    }
    log(f"\n  delta macro F1        : {f1_delta:+.6f}  (gate: >= {-MAX_F1_LOSS})")
    log(f"  delta per-label F1 sd : {spread_delta:+.6f}  (gate: < 0, lower is steadier)")

    out = BASE / "stratified_splits_results.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log("\n" + json.dumps(report, indent=2, default=float))
    log(f"\nWritten to {out.name}")
    if passes:
        log("\nPasses: `stratified_splits` may become a default.")
    elif spread_delta >= 0:
        log("\nFails on the thing it exists for: the per-label numbers are no steadier. "
            "Read the support line — with every label floored at min_samples and folds "
            "this large, a random draw may already be even enough to leave nothing to "
            "fix. A target with a longer tail is where this would show.")
    else:
        log("\nFails on quality: steadier, but the macro F1 paid for it.")


if __name__ == "__main__":
    main()
