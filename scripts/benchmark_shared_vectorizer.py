"""Is one vectorization pass instead of k worth the leakage it introduces?

Cross-validation refits the vectorizer per fold, so no fold's test rows influence its
vocabulary or IDF weights. That costs k passes over the corpus — 2.3 min each at 156 k
rows — plus one more for the deploy fit. Fitting ONE matrix over all rows and slicing it
per fold turns k+1 passes into one.

The price is real but small in kind: the vocabulary and the IDF statistics are computed
with the fold's test rows in view, so the out-of-fold metric carries a mild optimism.
The heads are still fit only on the training rows either way, so no row is ever *scored*
by a model that trained on it — the leak is in the feature weighting, not in the labels.

Whether that optimism is material at this scale is a measurement, not an opinion. This
script runs both modes on the same rows, the same folds and the same C grid, and reports
macro/micro F1 and wall-clock for each.

**Gate for adopting the shared matrix as the default: |delta macro F1| < 0.002 on both
targets.** A larger gap means the number would be flattering rather than faster.

Usage:
    python scripts/benchmark_shared_vectorizer.py [--rows N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from joblib import parallel_backend  # noqa: E402

from app.data import load_dataset, prepare_targets  # noqa: E402
from app.tuning import cross_val_evaluate  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
N_JOBS = 6
# `auto`'s shape, which is what a default change would affect (config.yaml).
K_FOLDS = 3
C_GRID = [2.0, 8.0, 32.0]
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"
UNIVERSITY = "http://w3id.org/openeduhub/vocabs/hochschulfaechersystematik/"

TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"
# Both targets the plan names, on the dataset the owner cleared for benchmark runs.
TARGETS = {
    "school subjects (discipline)": DISCIPLINE,
    "university subjects (hochschulfaechersystematik)": UNIVERSITY,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def run_mode(texts, y, classes, *, shared: bool) -> dict:
    """One cross-validation, either refitting per fold or slicing one matrix."""
    started = time.perf_counter()
    matrix = None
    if shared:
        matrix = TfidfBackend().fit_transform(texts)
    with parallel_backend("threading", n_jobs=N_JOBS):
        result = cross_val_evaluate(
            TfidfBackend, texts, y, classes,
            k=K_FOLDS, c_grid=C_GRID, seed=SEED, n_jobs=N_JOBS, matrix=matrix,
        )
    elapsed = time.perf_counter() - started
    best_c, _global_t, _per_label, metrics = result
    return {
        "seconds": round(elapsed, 1),
        "best_C": best_c,
        "f1_macro": metrics["f1_macro"],
        "f1_micro": metrics["f1_micro"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0,
                        help="cap the rows (0 = all) — for a quick shape check")
    args = parser.parse_args()

    report = {}
    for target, label_filter in TARGETS.items():
        log(f"\n=== {target} ===")
        data = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                            label_filter=label_filter)
        texts, label_lists = data.texts, data.label_lists
        if args.rows:
            texts, label_lists = texts[: args.rows], label_lists[: args.rows]
        y, classes, keep = prepare_targets(label_lists, min_samples=20)
        texts = [text for text, keeper in zip(texts, keep, strict=False) if keeper]
        log(f"{len(texts)} rows x {len(classes)} labels")
        if len(classes) < 2 or len(texts) < K_FOLDS:
            # Measured 2026-09-09: data_30k*.csv carries ONE university-labelled row, and
            # the 300 k export those models were trained from is not on this machine. Say
            # so and move on — a target that cannot be measured must not look measured.
            log("  SKIPPED: this dataset has no usable rows for that target.")
            report[target] = {"skipped": "no usable rows in this dataset"}
            continue

        refit = run_mode(texts, y, classes, shared=False)
        log(f"  refit per fold : {refit['seconds']:7.1f}s  macro {refit['f1_macro']:.4f}  "
            f"micro {refit['f1_micro']:.4f}  C={refit['best_C']}")
        shared = run_mode(texts, y, classes, shared=True)
        log(f"  shared matrix  : {shared['seconds']:7.1f}s  macro {shared['f1_macro']:.4f}  "
            f"micro {shared['f1_micro']:.4f}  C={shared['best_C']}")

        delta = shared["f1_macro"] - refit["f1_macro"]
        saved = 1 - shared["seconds"] / refit["seconds"] if refit["seconds"] else 0.0
        verdict = "PASSES the gate" if abs(delta) < 0.002 else "FAILS the gate"
        log(f"  delta macro F1 : {delta:+.4f}  ({verdict}: |delta| < 0.002)")
        log(f"  time saved     : {saved * 100:.1f}%")
        report[target] = {"refit": refit, "shared": shared,
                          "delta_macro_f1": round(delta, 5),
                          "time_saved_fraction": round(saved, 4),
                          "passes_gate": abs(delta) < 0.002}

    log("\n" + json.dumps(report, indent=2))
    measured = [entry for entry in report.values() if "passes_gate" in entry]
    if not measured:
        log("\nNothing was measured - the default stays as it is.")
    elif len(measured) < len(TARGETS):
        log("\nOnly part of the evidence. The gate asks for BOTH targets, so the default "
            "stays 'refit per fold' until the missing one can be measured.")
    elif all(entry["passes_gate"] for entry in measured):
        log("\nBoth targets pass: the shared matrix may become the default.")
    else:
        log("\nAt least one target fails: keep refitting per fold as the default.")


if __name__ == "__main__":
    main()
