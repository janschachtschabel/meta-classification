"""Does picking C on tuned thresholds produce a better model, or just a louder one?

`tuning.select_c` and `cross_val_evaluate` rank every C candidate at a flat 0.5 cut and
tune thresholds only on the winner. A candidate whose probabilities are ranked well but
scaled low is therefore eliminated before its thresholds exist.
`Profile.select_c_on_tuned_thresholds` scores each candidate under thresholds tuned for
itself instead, choosing the (C, thresholds) pair together. It costs no model fits — in
CV mode every candidate's out-of-fold probabilities are already in memory.

**Gate for making that the default (plan item C1): macro F1 up by >= 0.002 AND
`predicted_labels_per_row` no more than 10 % above the baseline.** The second half is
the owner's condition, and it is the point: a threshold rule can always buy macro F1 by
asserting more labels per row, and a model that answers "Mathematik, Physik, Chemie,
Informatik" to everything is a different product rather than a better one. Precision and
recall are reported beside them so the trade is visible rather than inferred.

**Measured on a held-out test split**, also at the owner's request. Both modes select C
and tune thresholds on a selection pool; the numbers that decide the gate come from rows
neither choice ever saw. The out-of-fold numbers are printed too, because the gap
between them is exactly the optimism that makes an in-sample comparison unsafe here: the
tuned rule fits one more decision to the data it is then scored on.

Usage:
    python scripts/benchmark_selection_rule.py [--dataset NAME] [--rows N]
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
from app.tuning import compute_metrics, cross_val_evaluate  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
# `auto`'s shape, which is what a default change would affect (config.yaml).
K_FOLDS = 3
C_GRID = [2.0, 8.0, 32.0]
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"

TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"

# The plan's gate, as two numbers rather than an opinion.
MIN_F1_GAIN = 0.002
MAX_EXTRA_LABELS = 0.10


def log(msg: str) -> None:
    print(msg, flush=True)


def run(texts, y, classes, pool, test, *, seed: int, on_tuned_thresholds: bool) -> dict:
    """Select C (+ thresholds) on the pool, then score the held-out test rows.

    The deploy fit mirrors what training actually ships: one head at the selected C over
    the whole pool, read through the thresholds that selection produced.
    """
    started = time.perf_counter()
    with parallel_backend("threading", n_jobs=N_JOBS):
        best_c, global_t, per_label_t, oof_metrics = cross_val_evaluate(
            TfidfBackend, [texts[i] for i in pool], y[pool], classes,
            k=K_FOLDS, c_grid=C_GRID, seed=seed, n_jobs=N_JOBS,
            select_on_tuned_thresholds=on_tuned_thresholds,
        )
        selection_seconds = time.perf_counter() - started

        vectorizer = TfidfBackend()
        x_pool = vectorizer.fit_transform([texts[i] for i in pool])
        head = make_head(best_c, n_jobs=N_JOBS, solver="newton-cg")
        head.fit(x_pool, y[pool])
        proba = head.predict_proba(vectorizer.transform([texts[i] for i in test]))
    metrics = compute_metrics(y[test], proba, classes, global_t, per_label_t)
    # 48 per-label entries per mode would bury the six numbers the gate reads.
    metrics.pop("per_label_f1", None)
    return {
        "selection_seconds": round(selection_seconds, 1),
        "best_C": best_c,
        "global_threshold": global_t,
        "held_out": metrics,
        "out_of_fold_f1_macro": oof_metrics["f1_macro"],
    }


def report_line(name: str, result: dict) -> str:
    m = result["held_out"]
    return (f"  {name:22}: macro {m['f1_macro']:.4f}  micro {m['f1_micro']:.4f}  "
            f"P {m['precision_macro']:.4f}  R {m['recall_macro']:.4f}  "
            f"labels/row {m['predicted_labels_per_row']:.3f}  "
            f"C={result['best_C']}  t={result['global_threshold']}  "
            f"{result['selection_seconds']:.0f}s")


def verdict(gain: float, same_best_c: bool) -> str:
    """Why the gate said no — the interesting half of a negative result."""
    if gain >= MIN_F1_GAIN:
        return ("Fails on over-assertion: the F1 was bought by claiming more labels per "
                "row, which is the failure mode this gate exists to catch. Keep the flat "
                "cut, or pair the rule with per-label shrinkage (plan item B4).")
    if same_best_c:
        return ("Fails on quality — and the reason is visible: the tuned rule picked the "
                "SAME C. On this target the flat 0.5 cut already ranks the candidates the "
                "way their tuned selves rank, so the rule had nothing to rescue. That is a "
                "property of this data, not a refutation of the mechanism: it would bite "
                "on a target whose candidates differ in CALIBRATION and not only in "
                "ranking. Re-measure there before dismissing it.")
    return ("Fails on quality: the rule moved the selection to a different C and the "
            "held-out model was not better for it. Keep the flat cut.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0,
                        help="cap the rows (0 = all) — for a quick shape check")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="draws BOTH the held-out split and the folds; re-run with "
                             "a second and a third to see whether a gain survives the draw")
    args = parser.parse_args()

    data = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                        label_filter=DISCIPLINE)
    texts, label_lists = data.texts, data.label_lists
    if args.rows:
        texts, label_lists = texts[: args.rows], label_lists[: args.rows]
    y, classes, keep = prepare_targets(label_lists, min_samples=20)
    texts = [text for text, keeper in zip(texts, keep, strict=False) if keeper]

    # Reuses the app's own splitter; train+val become the selection pool, so the test
    # rows are held out from BOTH the C search and the threshold tuning.
    train_idx, val_idx, test_idx = three_way_split(
        len(texts), val_size=0.2, test_size=0.2, seed=args.seed)
    pool = np.concatenate([train_idx, val_idx])
    log(f"school subjects (discipline): {len(texts)} rows x {len(classes)} labels, "
        f"{len(pool)} for selection, {len(test_idx)} held out, "
        f"{K_FOLDS} folds x {len(C_GRID)} candidates, seed {args.seed}")

    baseline = run(texts, y, classes, pool, test_idx,
                   seed=args.seed, on_tuned_thresholds=False)
    log(report_line("flat 0.5 cut (today)", baseline))
    tuned = run(texts, y, classes, pool, test_idx,
                seed=args.seed, on_tuned_thresholds=True)
    log(report_line("each candidate's own", tuned))

    base_m, tuned_m = baseline["held_out"], tuned["held_out"]
    gain = tuned_m["f1_macro"] - base_m["f1_macro"]
    extra = (tuned_m["predicted_labels_per_row"] / base_m["predicted_labels_per_row"]) - 1
    same_best_c = baseline["best_C"] == tuned["best_C"]
    passes = gain >= MIN_F1_GAIN and extra <= MAX_EXTRA_LABELS
    report = {
        "rows": len(texts), "labels": len(classes),
        "selection_rows": len(pool), "held_out_rows": len(test_idx),
        "baseline": baseline, "tuned": tuned,
        "delta_macro_f1": round(gain, 6),
        "extra_labels_per_row_fraction": round(extra, 4),
        "true_labels_per_row": base_m["true_labels_per_row"],
        "same_best_C": same_best_c,
        "passes_gate": passes,
    }

    log(f"\n  delta macro F1 : {gain:+.6f}  (gate: >= {MIN_F1_GAIN})")
    log(f"  labels per row : {base_m['predicted_labels_per_row']:.3f} -> "
        f"{tuned_m['predicted_labels_per_row']:.3f}  ({extra * 100:+.1f}%, gate: <= "
        f"{MAX_EXTRA_LABELS * 100:.0f}%; the data carries "
        f"{base_m['true_labels_per_row']:.3f})")
    log(f"  best_C         : {baseline['best_C']} -> {tuned['best_C']}  "
        f"({'same' if same_best_c else 'CHANGED'})")
    log(f"  out-of-fold    : {baseline['out_of_fold_f1_macro']:.4f} -> "
        f"{tuned['out_of_fold_f1_macro']:.4f}  (in-sample; the held-out line above is "
        f"what the gate reads)")

    report["seed"] = args.seed
    out = BASE / f"selection_rule_results_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log("\n" + json.dumps(report, indent=2, default=float))
    log(f"\nWritten to {out.name}")
    log("\nPasses: `select_c_on_tuned_thresholds` may become the default for `auto`."
        if passes else "\n" + verdict(gain, same_best_c))


if __name__ == "__main__":
    main()
