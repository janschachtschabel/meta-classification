"""Where does a training run's memory go? Peak and retained RSS per pipeline step.

A run on the full WLO export was OOM-killed on 2026-09-11. This measures each step of
the pipeline on its own, so a change that claims to lower the peak can be held to a
number (docs/plans/2026-09-11-training-memory.md holds the baseline and the gates).

Every step runs in a CHILD PROCESS of its own. Measured in one process, a step can
reuse memory an earlier step freed but never returned to the OS, and would read lower
than it is. "Peak" is the highest RSS above the RSS right before the step (sampled
every 5 ms), "retained" what is still held after the step's objects were deleted.

Steps:
  load              load_dataset on the whole file; retained = what the load keeps
  vectorize-before  word || char fitted in two threads with scikit-learn's own
                    fit_transform: the vectorizer as it was before 2026-09-11
  vectorize         TfidfBackend().fit_transform: what the app does today
  head-fit          one OneVsRest newton-cg fit (C=8) on --threads threads
  head-fit-budget   the same fit on the threads a ThreadBudget of (RSS + 3 x matrix)
                    grants; its absolute peak must stay under that budget (Phase 1 gate)
  identity          vectorize-before vs vectorize in one process: the same matrix,
                    array for array? Not a memory measurement.

Usage:
    python scripts/benchmark_training_memory.py [--dataset data_30k_ai.csv] [--rows N]
        [--threads 6] [--steps load,vectorize-before,vectorize,head-fit,...]
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
from joblib import parallel_backend  # noqa: E402
from scipy.sparse import hstack  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import prepare_targets  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.memory import MiB, PeakSampler, ThreadBudget, matrix_bytes, rss_bytes  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402
from app.vocabulary import count_terms  # noqa: E402

STEPS = ["load", "vectorize-before", "vectorize", "head-fit", "head-fit-budget", "identity"]
TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"
# The config.yaml default a real run applies to these columns.
WEIGHTS = {"properties.cclom:title": 2, "properties.cclom:general_keyword": 2}
MIN_SAMPLES = 20
C = 8.0


def settle() -> int:
    """RSS once the garbage is gone, so "retained" measures what is really kept."""
    gc.collect()
    time.sleep(0.2)
    return rss_bytes()


def load(args, *, truncate: bool = True) -> tuple[list[str], np.ndarray]:
    loaded = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                          label_filter=DISCIPLINE, text_column_weights=WEIGHTS)
    y, _classes, keep = prepare_targets(loaded.label_lists, MIN_SAMPLES)
    texts = [text for text, kept in zip(loaded.texts, keep, strict=False) if kept]
    if truncate and args.rows:
        texts, y = texts[: args.rows], y[: args.rows]
    return texts, y


def vectorize_before(texts: list[str]):
    """The pre-2026-09-11 TfidfBackend.fit_transform: both vocabularies at once."""
    backend = TfidfBackend()
    word = backend._make("word", backend.word_ngram, backend.max_word_features)
    char = backend._make("char_wb", backend.char_ngram, backend.max_char_features)
    with ThreadPoolExecutor(max_workers=2) as pool:
        word_future, char_future = pool.submit(word.fit_transform, texts), pool.submit(
            char.fit_transform, texts)
        matrix = hstack([word_future.result(), char_future.result()], format="csr")
    return (word, char), matrix


def measured(run) -> tuple[dict, object]:
    """Run ``run()`` under a 5 ms sampler; returns the timing/peak record and its result."""
    before = settle()
    started = time.perf_counter()
    with PeakSampler(interval=0.005) as sampler:
        result = run()
    record = {"seconds": round(time.perf_counter() - started, 1),
              "peak_mb": round((sampler.peak_bytes - before) / MiB),
              "peak_abs_mb": round(sampler.peak_bytes / MiB), "before_mb": round(before / MiB)}
    return record, result


def child(step: str, args) -> dict:
    if step == "load":
        # Never truncated: the file is read whole either way, and "retained" has to be
        # everything the load keeps, not the share a later step happens to use.
        record, (texts, y) = measured(lambda: load(args, truncate=False))
        record["retained_mb"] = round((settle() / MiB) - record["before_mb"])
        record["result"] = (f"{len(texts):,} rows x {y.shape[1]} labels, "
                            f"{sum(len(t.encode()) for t in texts) / MiB:.0f} MB of text")
        return record
    texts, y = load(args)
    if step in ("vectorize-before", "vectorize"):
        def vectorize():
            if step == "vectorize-before":
                return vectorize_before(texts)
            backend = TfidfBackend()
            return backend, backend.fit_transform(texts)

        record, (vectorizers, matrix) = measured(vectorize)
        record["result"] = f"{matrix_bytes(matrix) / MiB:.0f} MB matrix, nnz {matrix.nnz:,}"
        del vectorizers, matrix
        record["retained_mb"] = round(settle() / MiB) - record["before_mb"]
        return record
    if step in ("head-fit", "head-fit-budget"):
        matrix = TfidfBackend().fit_transform(texts)
        threads = args.threads
        if step == "head-fit-budget":
            budget = settle() + 3 * matrix_bytes(matrix)
            threads = ThreadBudget(requested=args.threads, budget_bytes=budget).for_matrix(matrix)
        head = make_head(C, n_jobs=threads, solver="newton-cg")

        def fit() -> None:
            with parallel_backend("threading", n_jobs=threads):
                head.fit(matrix, y)

        record, _ = measured(fit)
        record["result"] = (f"{threads} threads, {matrix_bytes(matrix) / MiB:.0f} MB matrix, "
                            f"peak {record['peak_mb'] / (matrix_bytes(matrix) / MiB):.1f}x matrix")
        if step == "head-fit-budget":
            record["budget_mb"] = round(budget / MiB)
            record["within_budget"] = record["peak_abs_mb"] <= record["budget_mb"]
        del head
        record["retained_mb"] = round(settle() / MiB) - record["before_mb"]
        return record
    if step == "identity":
        held_out = texts[-max(1, len(texts) // 10):]
        (word, char), reference = vectorize_before(texts)
        backend = TfidfBackend()
        current = backend.fit_transform(texts)
        def identical(a, b) -> bool:
            return (a.shape == b.shape and a.dtype == b.dtype
                    and all(np.array_equal(getattr(a, name), getattr(b, name))
                            and getattr(a, name).dtype == getattr(b, name).dtype
                            for name in ("data", "indices", "indptr")))

        same = identical(reference, current)
        same_fit = all(ref.vocabulary_ == cur.vocabulary_ and np.array_equal(ref.idf_, cur.idf_)
                       and ref.idf_.dtype == cur.idf_.dtype
                       for ref, cur in ((word, backend.word_vec), (char, backend.char_vec)))
        ref_t = hstack([word.transform(held_out), char.transform(held_out)], format="csr")
        same_t = identical(ref_t, backend.transform(held_out))
        # How far this corpus is from the two-pass fit's exactness limit (2**24).
        max_tf = max(int(count_terms(v.build_analyzer(), texts).term_frequencies().max())
                     for v in (word, char))
        return {"result": f"matrix identical={same}, vocabulary+idf identical={same_fit}, "
                          f"held-out transform identical={same_t}, "
                          f"largest term count {max_tf:,} (limit {2**24:,})",
                "identical": bool(same and same_fit and same_t)}
    raise SystemExit(f"unknown step {step!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0, help="cap the usable rows (0 = all)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--steps", default=",".join(STEPS[:5]))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(child(args.child, args)), flush=True)
        return

    print(f"{args.dataset}, rows={args.rows or 'all'}, threads={args.threads}", flush=True)
    rows = []
    for step in args.steps.split(","):
        command = [sys.executable, __file__, "--child", step, "--dataset", args.dataset,
                   "--rows", str(args.rows), "--threads", str(args.threads)]
        done = subprocess.run(command, capture_output=True, text=True)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if done.returncode or not lines:
            print(f"{step}: FAILED\n{done.stderr[-2000:]}", flush=True)
            continue
        record = {"step": step, **json.loads(lines[-1][len("RESULT "):])}
        rows.append(record)
        print(json.dumps(record), flush=True)

    print("\n| Step | Time | Peak | Retained | Result |\n|---|---:|---:|---:|---|")
    for r in rows:
        peak = f"+{r['peak_mb']:,} MB" if "peak_mb" in r else "–"
        kept = f"{r['retained_mb']:+,} MB" if "retained_mb" in r else "–"
        seconds = f"{r['seconds']} s" if "seconds" in r else "–"
        extra = f" — budget {r['budget_mb']:,} MB, within: {r['within_budget']}" if (
            "budget_mb" in r) else ""
        print(f"| {r['step']} | {seconds} | {peak} | {kept} | {r['result']}{extra} |")


if __name__ == "__main__":
    main()
