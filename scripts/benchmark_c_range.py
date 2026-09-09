"""How far up does the C grid actually need to reach?

Four consecutive trainings selected the grid MAXIMUM (16 -> 32 -> 128), each widening
gaining real F1, so the upper endpoint has never been tested to exhaustion. This walks C
upward on a holdout split -- one fit per candidate instead of k x |grid| -- and reports
where the curve stops paying.

Measures three things per candidate, because they can disagree:
  - macro F1 at the 0.5 cut: EXACTLY the criterion `tuning.select_c` uses, so it predicts
    which C a real run would pick.
  - macro F1 with tuned thresholds: the quality actually reported. Tuned and scored on the
    same split, so optimistic in absolute terms -- read the SHAPE, not the level.
  - convergence warnings: a high C that wins without converging has not won.

Usage (from the repo root):
    python scripts/benchmark_c_range.py university
    python scripts/benchmark_c_range.py school --grid 32,128,512,2048
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.classifier import make_head  # noqa: E402
from app.prepare import prepare_data  # noqa: E402
from app.profiles import load_training_config  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.thresholds import macro_f1, tune_thresholds  # noqa: E402
from app.tuning import _default_decision, compute_metrics  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

VOCABS = {
    "school": "http://w3id.org/openeduhub/vocabs/discipline/",
    "university": "http://w3id.org/openeduhub/vocabs/hochschulfaechersystematik/",
}
DATA = Path(r"C:\Users\jan\staging\Windsurf\classifikation-api-lightgbm\data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(VOCABS))
    parser.add_argument("--grid", default="8,32,128,512,2048")
    parser.add_argument("--dataset", default="data_300k.csv")
    args = parser.parse_args()
    grid = [float(part) for part in args.grid.split(",")]

    settings = Settings(data_dir=DATA, models_dir=_REPO_ROOT / "models", auth_enabled=False)
    training_cfg = load_training_config(_REPO_ROOT / "config.yaml")
    req = {
        "dataset_name": args.dataset,
        "model_name": "_c_range_probe",
        "text_columns": [
            "properties.cclom:title",
            "properties.cclom:general_description",
            "properties.cclom:general_keyword",
        ],
        "label_column": "properties.ccm:taxonid",
        "csv_separator": ";",
        "label_separator": ",",
        "label_filter": VOCABS[args.target],
        "min_samples_per_label": 20,
    }

    t0 = time.time()
    prep = prepare_data(req, settings, training_cfg, cv_folds=0,
                        on_progress=lambda **_: None, should_stop=lambda: False)
    assert prep is not None
    print(f"{args.target}: {len(prep.texts)} rows, {len(prep.classes)} labels, "
          f"task={prep.task_type}  (prepared in {(time.time() - t0) / 60:.1f} min)", flush=True)

    # One vectorization for the whole sweep: only C changes between candidates.
    vectorizer = TfidfBackend()
    x_tr = vectorizer.fit_transform(prep.texts[prep.train_idx].tolist())
    x_va = vectorizer.transform(prep.texts[prep.val_idx].tolist())
    y_tr, y_va = prep.y_all[prep.train_idx], prep.y_all[prep.val_idx]
    print(f"train matrix: {x_tr.shape[0]}x{x_tr.shape[1]}, nnz={x_tr.nnz:,}"
          f"  (features ready at {(time.time() - t0) / 60:.1f} min)\n", flush=True)
    print(f"{'C':>8} {'macro@0.5':>10} {'macro tuned':>12} {'micro tuned':>12} "
          f"{'fit min':>8} {'conv.warn':>10}", flush=True)

    n_jobs = settings.effective_n_jobs()
    for c in grid:
        started = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            head = make_head(c, n_jobs=n_jobs, solver=settings.solver)
            head.fit(x_tr, y_tr)
            proba = head.predict_proba(x_va)
        fit_min = (time.time() - started) / 60
        selection = macro_f1(y_va, _default_decision(proba, prep.task_type))
        global_t, per_label = tune_thresholds(y_va, proba, prep.classes, per_label=True)
        tuned = compute_metrics(y_va, proba, prep.classes, global_t, per_label,
                                task_type=prep.task_type)
        print(f"{c:>8.0f} {selection:>10.4f} {tuned['f1_macro']:>12.4f} "
              f"{tuned['f1_micro']:>12.4f} {fit_min:>8.1f} "
              f"{sum(1 for w in caught if issubclass(w.category, ConvergenceWarning)):>10}",
              flush=True)
        del head, proba

    print(f"\ntotal {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
