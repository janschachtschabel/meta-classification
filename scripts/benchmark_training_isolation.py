"""What the API process keeps after two trainings: run in a thread vs in a child process.

Task 4.4 of docs/plans/2026-09-11-training-memory.md. Each mode runs in a fresh process
of its own that stands in for the API process. It runs two trainings back to back on a
thread, as the job runner does — the same dataset under two model names, a queue of
runs, the case that thrashed in July — and reads its own RSS at the start, after each
run, and once both models are loaded for serving. In thread mode the models are already
in the cache; in process mode this process loads them from disk, which is what the first
/predict of each would do. The difference in that last column is what the trainings
left behind in the process that serves requests.

"API peak" is this process' highest RSS during a run (sampled every 50 ms), "training
peak" the run's own reported peak — in process mode the child's.

Usage:
    python scripts/benchmark_training_isolation.py [--dataset data_30k_ai.csv]
        [--data-dir DIR] [--profile auto] [--modes thread,process]
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.memory import MiB, PeakSampler, rss_bytes  # noqa: E402
from app.profiles import load_training_config  # noqa: E402
from app.registry import Registry  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.train_worker import run_in_child  # noqa: E402
from app.training import run_training  # noqa: E402

MODES = ["thread", "process"]
TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"


def settled_mb() -> int:
    """RSS once the garbage is gone, so the reading is what is really kept."""
    gc.collect()
    time.sleep(0.5)
    return rss_bytes() // MiB


def train_on_a_thread(target, req: dict, settings: Settings, config, profile,
                      registry: Registry) -> dict:
    """One run the way the job runner starts it: on a thread of this process."""
    reported = [0]
    failure: list[BaseException] = []

    def on_progress(**fields) -> None:
        reported[0] = max(reported[0], int(fields.get("peak_rss_mb") or 0))

    def run() -> None:
        try:
            target(req, settings, config, profile, registry, on_progress=on_progress,
                   should_stop=lambda: False)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            failure.append(exc)

    started = time.perf_counter()
    with PeakSampler(interval=0.05) as sampler:
        thread = threading.Thread(target=run, name="training")
        thread.start()
        thread.join()
    if failure:
        raise failure[0]
    return {"seconds": round(time.perf_counter() - started, 1),
            "api_peak_mb": sampler.peak_bytes // MiB, "training_peak_mb": reported[0],
            "after_mb": settled_mb()}


def child(mode: str, args) -> dict:
    models_dir = Path(tempfile.mkdtemp(prefix=f"apiv3-isolation-{mode}-"))
    try:
        settings = Settings(data_dir=Path(args.data_dir), models_dir=models_dir,
                            auth_enabled=False)
        config = load_training_config(settings.config_file)
        profile = config.get(args.profile)
        registry = Registry(models_dir, settings.effective_max_models_in_memory())
        target = run_training if mode == "thread" else run_in_child
        record: dict = {"mode": mode, "start_mb": settled_mb(), "runs": []}
        names = [f"isolation_{mode}_{run}" for run in (1, 2)]
        for name in names:
            req = {"dataset_name": args.dataset, "model_name": name,
                   "text_columns": TEXT_COLUMNS, "label_column": LABEL_COLUMN,
                   "csv_separator": ";", "label_separator": ",", "label_filter": DISCIPLINE}
            record["runs"].append(train_on_a_thread(target, req, settings, config, profile,
                                                    registry))
        for name in names:
            registry.get(name)
        record["serving_mb"] = settled_mb()
        return record
    finally:
        shutil.rmtree(models_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--data-dir", default=str(BASE / "data"))
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(child(args.child, args)), flush=True)
        return

    print(f"{args.dataset}, profile {args.profile}, two runs per mode", flush=True)
    rows = []
    for mode in args.modes.split(","):
        command = [sys.executable, __file__, "--child", mode, "--dataset", args.dataset,
                   "--data-dir", args.data_dir, "--profile", args.profile]
        done = subprocess.run(command, capture_output=True, text=True)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if done.returncode or not lines:
            print(f"{mode}: FAILED\n{done.stderr[-2000:]}", flush=True)
            continue
        record = json.loads(lines[-1][len("RESULT "):])
        rows.append(record)
        print(json.dumps(record), flush=True)

    print("\n| Mode | API process at start | after run 1 | after run 2 | serving both models "
          "| API peak (run 1 / 2) | training peak (run 1 / 2) | time (run 1 / 2) |"
          "\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        one, two = r["runs"]
        print(f"| {r['mode']} | {r['start_mb']:,} MB | {one['after_mb']:,} MB "
              f"| {two['after_mb']:,} MB | {r['serving_mb']:,} MB "
              f"| {one['api_peak_mb']:,} / {two['api_peak_mb']:,} MB "
              f"| {one['training_peak_mb']:,} / {two['training_peak_mb']:,} MB "
              f"| {one['seconds']} / {two['seconds']} s |")


if __name__ == "__main__":
    main()
