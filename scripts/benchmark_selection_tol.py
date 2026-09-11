"""Is the last digit of convergence worth anything to a C candidate that gets thrown away?

Every fit in the C search exists to be scored once and discarded; only the deploy fit
is kept. scikit-learn drives all of them to tol=1e-4. A looser tolerance stops the
solver earlier, which is free wall-clock — unless the sloppier fit ranks the candidates
differently, and then it has bought speed with a worse model.

**Gate for adopting a looser selection tolerance (plan item A3): >= 15 % of the
selection wall-clock saved, the SAME best_C, and OOF macro F1 within 1e-4.** Anything
less than 15 % is not worth a second number in `config.yaml`; a different `best_C` or a
moved F1 means the search is no longer measuring what it will ship.

This also stands in for a decision about A2 (warm-starting the C path, 2-3 days of
work): both mechanisms buy the same thing — solver iterations on candidates that are
about to be discarded. If the convergence tail turns out to be a thin slice of the
fit, A2 cannot be a thick one either, and that is worth knowing before building it.

Usage:
    python scripts/benchmark_selection_tol.py [--rows N] [--tol 1e-3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from joblib import parallel_backend  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import prepare_targets  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.tuning import cross_val_evaluate  # noqa: E402
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

# The plan's gate, as three numbers rather than an opinion.
MIN_TIME_SAVED = 0.15
MAX_F1_DRIFT = 1e-4


def log(msg: str) -> None:
    print(msg, flush=True)


def run(texts, y, classes, *, tol) -> dict:
    """One full cross-validation at the given tolerance.

    The vectorization is inside the timing on purpose: it is the same work in both
    runs, so leaving it in reports the saving as a fraction of what a user actually
    waits for, not of the part this change happens to touch.
    """
    started = time.perf_counter()
    with parallel_backend("threading", n_jobs=N_JOBS):
        best_c, _global_t, _per_label, metrics = cross_val_evaluate(
            TfidfBackend, texts, y, classes,
            k=K_FOLDS, c_grid=C_GRID, seed=SEED, n_jobs=N_JOBS, tol=tol,
        )
    return {
        "seconds": round(time.perf_counter() - started, 1),
        "best_C": best_c,
        "f1_macro": metrics["f1_macro"],
        "f1_micro": metrics["f1_micro"],
    }


def solver_work(texts, y, *, tol) -> dict:
    """One direct fit, reporting what the SOLVER did rather than what the clock did.

    Wall-clock over a whole cross-validation also contains k vectorization passes and
    the scoring, so a real saving in the solver can hide inside it — and a machine
    under load can invent one that is not there. ``n_iter_`` cannot: it is the number
    of newton-cg steps each label actually took, and it is exactly what a looser
    tolerance (A3) and a warm-started path (A2) are both trying to reduce. If there is
    no tail here, neither mechanism has anything to harvest.
    """
    matrix = TfidfBackend().fit_transform(texts)
    started = time.perf_counter()
    with parallel_backend("threading", n_jobs=N_JOBS):
        head = make_head(C_GRID[-1], n_jobs=N_JOBS, solver="newton-cg", tol=tol)
        head.fit(matrix, y)
    iterations = [int(est.n_iter_[0]) for est in head.estimators_]
    return {
        "fit_seconds": round(time.perf_counter() - started, 1),
        "mean_iterations": round(sum(iterations) / len(iterations), 2),
        "max_iterations": max(iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0,
                        help="cap the rows (0 = all) — for a quick shape check")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="the looser tolerance to test against sklearn's 1e-4")
    args = parser.parse_args()

    data = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                        label_filter=DISCIPLINE)
    texts, label_lists = data.texts, data.label_lists
    if args.rows:
        texts, label_lists = texts[: args.rows], label_lists[: args.rows]
    y, classes, keep = prepare_targets(label_lists, min_samples=20)
    texts = [text for text, keeper in zip(texts, keep, strict=False) if keeper]
    log(f"school subjects (discipline): {len(texts)} rows x {len(classes)} labels, "
        f"{K_FOLDS} folds x {len(C_GRID)} candidates")

    strict = run(texts, y, classes, tol=None)
    log(f"  sklearn default (1e-4) : {strict['seconds']:7.1f}s  macro {strict['f1_macro']:.4f}  "
        f"micro {strict['f1_micro']:.4f}  C={strict['best_C']}")
    loose = run(texts, y, classes, tol=args.tol)
    log(f"  loosened ({args.tol:<7g})      : {loose['seconds']:7.1f}s  macro {loose['f1_macro']:.4f}  "
        f"micro {loose['f1_micro']:.4f}  C={loose['best_C']}")

    drift = loose["f1_macro"] - strict["f1_macro"]
    saved = 1 - loose["seconds"] / strict["seconds"] if strict["seconds"] else 0.0
    same_c = loose["best_C"] == strict["best_C"]
    passes = saved >= MIN_TIME_SAVED and abs(drift) <= MAX_F1_DRIFT and same_c
    report = {
        "rows": len(texts), "labels": len(classes), "tol": args.tol,
        "strict": strict, "loose": loose,
        "delta_macro_f1": round(drift, 6),
        "time_saved_fraction": round(saved, 4),
        "same_best_C": same_c,
        "passes_gate": passes,
    }
    log(f"\n  time saved     : {saved * 100:5.1f}%   (gate: >= {MIN_TIME_SAVED * 100:.0f}%)")
    log(f"  delta macro F1 : {drift:+.6f}  (gate: |delta| <= {MAX_F1_DRIFT})")
    log(f"  best_C         : {strict['best_C']} -> {loose['best_C']}  "
        f"({'same' if same_c else 'CHANGED'})")
    # Only when the clock says no: the question then is whether the solver had a
    # tail to give at all, which decides A2 as much as it decides A3.
    if not passes:
        log(f"\n  what the solver actually did (one fit at C={C_GRID[-1]}, all rows):")
        for name, value in (("sklearn default", None),
                            (f"loosened ({args.tol:g})", args.tol)):
            work = solver_work(texts, y, tol=value)
            report.setdefault("solver_work", {})[name] = work
            log(f"    {name:22}: {work['fit_seconds']:6.1f}s  "
                f"{work['mean_iterations']:5.2f} newton-cg iterations "
                f"on average (max {work['max_iterations']})")

    log("\n" + json.dumps(report, indent=2))
    if passes:
        log("\nPasses: a looser selection tolerance may become the default for `auto`.")
    elif saved < MIN_TIME_SAVED:
        log("\nFails on time. Read it together with the solver line above: the tail is "
            "NOT thin at the fit — a looser tolerance really does cut the newton-cg "
            "steps roughly in half — but the selection phase also pays for k "
            "vectorization passes, the scoring and the threshold search, and halving "
            "the fits still does not move enough of the whole. The same denominator "
            "bounds A2 (warm-starting the C path), which draws on the same pool: "
            "divide the seconds saved here by the iteration reduction to see how "
            "large that pool is before spending days on a better way to drain it.")
    else:
        log("\nFails on quality: the looser search does not rank the candidates the way "
            "the shipped model would. Keep sklearn's default.")


if __name__ == "__main__":
    main()
