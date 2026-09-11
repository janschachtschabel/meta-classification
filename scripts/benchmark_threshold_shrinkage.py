"""Is a per-label threshold fitted on four positives worth trusting?

`tune_threshold_columns` picks each label's cut by F1-argmax over the validation
positives it happens to have. With 48 labels that number ranges from a handful to
several hundred on the same run, and the rule treats both the same.
`Profile.threshold_shrinkage_k` blends each cut toward the GLOBAL one with weight
`n_pos / (n_pos + k)`, so a well-supported label keeps its own answer and a sparse one
mostly borrows.

**Gate for making it a default (plan item B4): macro F1 up by >= 0.002 AND
`predicted_labels_per_row` no more than 10 % above the baseline** — the same two
conditions C1 was held to, because it is the same decision being changed. Precision and
recall are printed beside them.

**k = 10 is chosen a priori, not tuned here.** It was fixed from the HOLDOUT path's
support distribution (median ~107 validation positives, 6 of 48 labels under 20), where
it leaves a typical label 91 % of its own cut. On the CV path measured below the pool is
four times larger — median ~420 positives, 2 of 48 under 20 — so the same k leaves a
typical label 97.7 % of its cut and bites almost only on the sparse tail. That is the
honest reading of the script's own support line, and it is why the effect here is a tail
effect rather than a broad one. Fitting k against the held-out split the gate reads would
make the result meaningless, so k is fixed and only the SEED varies.

Sibling result worth knowing before reading this one: plan item C2 proposed the opposite
move, replacing the 0.05 grid with every observed score as a candidate. Measured on this
same target, that is worth **-0.0115** macro F1 held out while being **+0.0037**
in-sample — a finer cut fits the validation split's noise. Shrinkage is the correction
for that failure, not an addition to it.

Usage:
    python scripts/benchmark_threshold_shrinkage.py [--seed N] [--shrink-k K] [--rows N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
from joblib import parallel_backend  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import prepare_targets, three_way_split  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.profiles import load_training_config  # noqa: E402
from app.settings import get_settings  # noqa: E402
from app.tuning import compute_metrics, cross_val_evaluate  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
K_FOLDS = 3
C_GRID = [2.0, 8.0, 32.0]
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"

TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"

MIN_F1_GAIN = 0.002
MAX_EXTRA_LABELS = 0.10

# Read from the SHIPPED profile rather than hardcoded, and deliberately so: this
# benchmark's baseline has to be what a training actually does today, or the delta
# credits B4 with a gain that C1 already banked. A first run of this script left it at
# the function default (False) and measured shrinkage on top of the pre-C1 rule --
# the exact double-count the plan warns about for C1/C2/B4.
_SHIPPED = load_training_config(get_settings().config_file).get("auto")
SELECT_ON_TUNED = _SHIPPED.select_c_on_tuned_thresholds


def log(msg: str) -> None:
    print(msg, flush=True)


def run(texts, y, classes, pool, test, *, seed: int, shrink_k: float | None) -> dict:
    """Select C + thresholds on the pool, then score the held-out rows.

    Shrinkage changes no model fit — only how the cuts are read off the out-of-fold
    probabilities — but it runs through the real pipeline anyway so that the number
    describes what a training would actually deploy.
    """
    started = time.perf_counter()
    with parallel_backend("threading", n_jobs=N_JOBS):
        best_c, global_t, per_label_t, _oof = cross_val_evaluate(
            TfidfBackend, [texts[i] for i in pool], y[pool], classes,
            k=K_FOLDS, c_grid=C_GRID, seed=seed, n_jobs=N_JOBS,
            select_on_tuned_thresholds=SELECT_ON_TUNED,
            threshold_shrink_k=shrink_k,
        )
        selection_seconds = time.perf_counter() - started

        vectorizer = TfidfBackend()
        x_pool = vectorizer.fit_transform([texts[i] for i in pool])
        head = make_head(best_c, n_jobs=N_JOBS, solver="newton-cg")
        head.fit(x_pool, y[pool])
        proba = head.predict_proba(vectorizer.transform([texts[i] for i in test]))
    metrics = compute_metrics(y[test], proba, classes, global_t, per_label_t)
    metrics.pop("per_label_f1", None)
    cuts = np.array([per_label_t[uri] for uri in classes])
    return {
        "selection_seconds": round(selection_seconds, 1),
        "best_C": best_c,
        "global_threshold": global_t,
        "held_out": metrics,
        "threshold_spread": round(float(cuts.std()), 4),
        "threshold_min": round(float(cuts.min()), 4),
        "threshold_max": round(float(cuts.max()), 4),
    }


def report_line(name: str, result: dict) -> str:
    m = result["held_out"]
    return (f"  {name:26}: macro {m['f1_macro']:.4f}  micro {m['f1_micro']:.4f}  "
            f"P {m['precision_macro']:.4f}  R {m['recall_macro']:.4f}  "
            f"labels/row {m['predicted_labels_per_row']:.3f}  C={result['best_C']}  "
            f"cuts {result['threshold_min']:.2f}-{result['threshold_max']:.2f} "
            f"(sd {result['threshold_spread']:.3f})  {result['selection_seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0,
                        help="cap the rows (0 = all) — for a quick shape check")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="draws BOTH the held-out split and the folds")
    parser.add_argument("--shrink-k", type=float, default=10.0,
                        help="positives at which a label's own cut carries half the "
                             "weight; fixed a priori, see the module docstring")
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
    pool = np.concatenate([train_idx, val_idx])
    support = y[pool].sum(axis=0)
    log(f"school subjects (discipline): {len(texts)} rows x {len(classes)} labels, "
        f"{len(pool)} for selection, {len(test_idx)} held out, seed {args.seed}, "
        f"C picked on tuned thresholds: {SELECT_ON_TUNED}")
    log(f"  label support in the pool: min {int(support.min())}, "
        f"median {int(np.median(support))}, max {int(support.max())}; "
        f"{int((support < 20).sum())} labels under 20")

    baseline = run(texts, y, classes, pool, test_idx, seed=args.seed, shrink_k=None)
    log(report_line("own cut, unshrunk", baseline))
    shrunk = run(texts, y, classes, pool, test_idx, seed=args.seed, shrink_k=args.shrink_k)
    log(report_line(f"shrunk toward global k={args.shrink_k:g}", shrunk))

    base_m, shrunk_m = baseline["held_out"], shrunk["held_out"]
    gain = shrunk_m["f1_macro"] - base_m["f1_macro"]
    extra = (shrunk_m["predicted_labels_per_row"] / base_m["predicted_labels_per_row"]) - 1
    passes = gain >= MIN_F1_GAIN and extra <= MAX_EXTRA_LABELS
    report = {
        "rows": len(texts), "labels": len(classes), "seed": args.seed,
        "shrink_k": args.shrink_k,
        "selection_rows": len(pool), "held_out_rows": len(test_idx),
        "labels_under_20_positives": int((support < 20).sum()),
        "baseline": baseline, "shrunk": shrunk,
        "delta_macro_f1": round(gain, 6),
        "extra_labels_per_row_fraction": round(extra, 4),
        "true_labels_per_row": base_m["true_labels_per_row"],
        "same_best_C": baseline["best_C"] == shrunk["best_C"],
        "passes_gate": passes,
    }

    log(f"\n  delta macro F1 : {gain:+.6f}  (gate: >= {MIN_F1_GAIN})")
    log(f"  labels per row : {base_m['predicted_labels_per_row']:.3f} -> "
        f"{shrunk_m['predicted_labels_per_row']:.3f}  ({extra * 100:+.1f}%, gate: <= "
        f"{MAX_EXTRA_LABELS * 100:.0f}%; the data carries "
        f"{base_m['true_labels_per_row']:.3f})")
    log(f"  cut spread     : sd {baseline['threshold_spread']:.3f} -> "
        f"{shrunk['threshold_spread']:.3f}  (shrinkage must reduce it, or it did nothing)")

    out = BASE / f"threshold_shrinkage_results_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log("\n" + json.dumps(report, indent=2, default=float))
    log(f"\nWritten to {out.name}")
    if passes:
        log("\nPasses: `threshold_shrinkage_k` may become a default.")
    elif gain < MIN_F1_GAIN:
        log("\nFails on quality: borrowing from the global cut did not buy a better "
            "held-out model here. Check the support line above — a target whose labels "
            "all have hundreds of positives has nothing for this to fix.")
    else:
        log("\nFails on over-assertion: the F1 came with more labels per row, which is "
            "a different product rather than a better model.")


if __name__ == "__main__":
    main()
