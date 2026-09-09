"""Is the 0.05 threshold grid too coarse, or exactly coarse enough?

`thresholds._DEFAULT_GRID` offers 19 fixed cuts. Plan item C2 proposed replacing it with
every observed score as a candidate: for one label, the F1-optimal cut always sits at
some observed score, and a single descending sweep finds it exactly in O(n log n). The
grid can only ever approximate that.

**The catch is that a finer search is better on the split it is tuned on BY
CONSTRUCTION**, so tuning and scoring on the same rows would prove nothing whatever the
number said. This script tunes on a validation split and reads the result on a held-out
one, and reports both — the in-sample column is the correctness check on the sweep (a
search over every cut cannot lose to a search over 19 of them on the rows both saw; if
it does, the sweep is buggy and the held-out comparison is void).

Fixed C on purpose: this asks about the threshold rule alone, so the C search is out of
the picture rather than confounding it.

Usage:
    python scripts/benchmark_threshold_grid.py [--seed N] [--c 32] [--rows N]
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
from joblib import parallel_backend  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import load_dataset, prepare_targets, three_way_split  # noqa: E402
from app.thresholds import macro_f1, tune_threshold_columns  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"
TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"


def log(msg: str) -> None:
    print(msg, flush=True)


def curve_columns(y_val: np.ndarray, proba: np.ndarray, global_t: float) -> np.ndarray:
    """Plan item C2: every observed score is a candidate, not just the 19 on the grid.

    Deliberately lives here and not in ``app``: measured below it is WORSE out of sample
    than the coarse grid it would replace, so there is nothing to ship. Kept so the
    claim can be re-checked on another target instead of taken on trust.
    """
    columns = np.full(proba.shape[1], global_t, dtype=float)
    for col in range(proba.shape[1]):
        truth = y_val[:, col]
        if truth.sum() == 0:
            continue
        order = np.argsort(-proba[:, col])
        scores, hits = proba[order, col], truth[order]
        tp = np.cumsum(hits)
        fp = np.cumsum(1 - hits)
        fn = hits.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
        # A cut cannot separate rows that share a score, so only positions where the
        # next score differs are reachable thresholds.
        f1 = np.where(np.r_[scores[1:] != scores[:-1], True], f1, -1.0)
        columns[col] = float(scores[int(np.argmax(f1))])
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--c", type=float, default=32.0,
                        help="fixed regularization; the C search is not the question here")
    args = parser.parse_args()

    data = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                        label_filter=DISCIPLINE)
    texts, label_lists = data.texts, data.label_lists
    if args.rows:
        texts, label_lists = texts[: args.rows], label_lists[: args.rows]
    y, classes, keep = prepare_targets(label_lists, min_samples=20)
    texts = [text for text, keeper in zip(texts, keep, strict=False) if keeper]

    train_idx, val_idx, test_idx = three_way_split(
        len(texts), val_size=0.2, test_size=0.2, seed=args.seed)
    log(f"school subjects (discipline): {len(texts)} rows x {len(classes)} labels; "
        f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}, "
        f"seed {args.seed}, C={args.c:g}")

    vectorizer = TfidfBackend()
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    with parallel_backend("threading", n_jobs=N_JOBS):
        head = make_head(args.c, n_jobs=N_JOBS, solver="newton-cg")
        head.fit(x_train, y[train_idx])
        p_val = head.predict_proba(vectorizer.transform([texts[i] for i in val_idx]))
        p_test = head.predict_proba(vectorizer.transform([texts[i] for i in test_idx]))
    y_val, y_test = y[val_idx], y[test_idx]

    global_t, grid_cuts = tune_threshold_columns(y_val, p_val, per_label=True)
    curve_cuts = curve_columns(y_val, p_val, global_t)

    scores = {}
    for name, cuts in (("0.05 grid (today)", grid_cuts), ("every observed cut (C2)", curve_cuts)):
        scores[name] = {
            "in_sample": macro_f1(y_val, (p_val >= cuts).astype(int)),
            "held_out": macro_f1(y_test, (p_test >= cuts).astype(int)),
            "labels_per_row": round(float((p_test >= cuts).sum(axis=1).mean()), 3),
        }
        s = scores[name]
        log(f"  {name:26}: in-sample {s['in_sample']:.4f}   held out {s['held_out']:.4f}   "
            f"labels/row {s['labels_per_row']:.3f}")

    grid, curve = scores["0.05 grid (today)"], scores["every observed cut (C2)"]
    sweep_ok = curve["in_sample"] >= grid["in_sample"]
    report = {
        "rows": len(texts), "labels": len(classes), "seed": args.seed, "C": args.c,
        "global_threshold": global_t, "scores": scores,
        "delta_in_sample": round(curve["in_sample"] - grid["in_sample"], 6),
        "delta_held_out": round(curve["held_out"] - grid["held_out"], 6),
        "sweep_beats_grid_in_sample": sweep_ok,
        "mean_abs_cut_difference": round(float(np.abs(curve_cuts - grid_cuts).mean()), 4),
    }
    log(f"\n  delta in-sample : {report['delta_in_sample']:+.6f}  "
        f"(must be >= 0 or the sweep is buggy: {sweep_ok})")
    log(f"  delta held out  : {report['delta_held_out']:+.6f}")
    log(f"  mean |cut difference|: {report['mean_abs_cut_difference']:.4f}")

    out = BASE / f"threshold_grid_results_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log("\n" + json.dumps(report, indent=2, default=float))
    log(f"\nWritten to {out.name}")
    if not sweep_ok:
        log("\nINVALID: the exhaustive sweep lost to the grid on the rows both were "
            "tuned on, which cannot happen. Fix the sweep before reading anything else.")
    elif report["delta_held_out"] < 0:
        log("\nKeep the coarse grid. The finer search wins where it is fitted and loses "
            "where it counts: the extra resolution buys the tuning split's noise. "
            "Shrinking a cut toward the global one (plan item B4) is the correction for "
            "that failure — measured separately in benchmark_threshold_shrinkage.py.")
    else:
        log("\nThe finer search survives out of sample here; worth re-checking on a "
            "second target before changing the grid.")


if __name__ == "__main__":
    main()
