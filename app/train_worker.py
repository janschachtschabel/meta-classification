"""Run one training in a child process: a JSON job spec in, JSON lines out.

A training in a thread of the API process leaves its memory with the process that serves
requests (the allocator returns little of it to the OS), and an OOM kill takes the API
down with it. In a child process every byte goes back when the run ends, and a kill ends
the run, not the server: the job fails with a reason instead of the container restarting.

``subprocess`` and JSON rather than ``multiprocessing``: nothing is pickled, and the
same code runs on Windows and Linux. The protocol, one JSON object per line:

    stdin  -> {"job": {...}}         the first line: request, settings, config, profile
    stdin  -> {"stop": true}         later: stop at the next checkpoint
    stdout <- {"progress": {...}}    every progress update of the run
    stdout <- {"done": {"result": {...}, "staged": true|false}}
    stdout <- {"failed": {"message": "...", "user_facing": true|false}}

The child writes the bundle into the registry's hidden staging directory only; the
parent publishes it under its OWN disk lock, so the one-lock rule of ``registry.py``
holds across the process boundary. End of stdin means the parent is gone or wants the
run dead: the child exits at once. Its log goes to the inherited stderr.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from .classifier import ClassifierModel
from .errors import TrainingInputError, TrainingProcessError, UserFacingError
from .memory import share_budget_with
from .profiles import Profile, TrainingConfig
from .registry import Registry
from .settings import Settings

logger = logging.getLogger("api_v3.jobs")

# Never given to the child, neither in the spec nor through its environment: it trains
# and saves, it authenticates nobody.
SECRET_SETTINGS = frozenset({"api_key_admin", "api_key_readonly"})
_SECRET_ENV = frozenset(f"APIV3_{name.upper()}" for name in SECRET_SETTINGS)
# How long a stopped child may take to exit before it is killed outright.
_EXIT_GRACE_SECONDS = 10
# Linux: how willing the kernel's OOM killer is to pick this process, -1000..1000.
_OOM_SCORE_ADJ = Path("/proc/self/oom_score_adj")
# A child killed by SIGKILL: Popen reports -9; through a shell it would be 128 + 9.
_KILLED_BY_SIGNAL_9 = (-9, 137)
# Where `python -m app.train_worker` resolves the package from.
_APP_ROOT = Path(__file__).resolve().parent.parent


def job_spec(req: dict, settings: Settings, training_cfg: TrainingConfig, profile: Profile) -> dict:
    """Everything a child needs to run ``run_training`` as the parent would have."""
    settings_doc = settings.model_dump(mode="json", exclude=set(SECRET_SETTINGS))
    # A relative path means this process' working directory; the child runs in the
    # package root, where the same string would name another place.
    for name, value in settings:
        if isinstance(value, Path) and name in settings_doc:
            settings_doc[name] = str(value.resolve())
    return {
        "req": req,
        "settings": settings_doc,
        "training_config": dataclasses.asdict(training_cfg),
        "profile": dataclasses.asdict(profile),
        # The child's memory budget covers this process too: the limit is the container's.
        "parent_pid": os.getpid(),
    }


def read_job(spec: dict) -> tuple[dict, Settings, TrainingConfig, Profile]:
    """The inverse of :func:`job_spec`, in the child."""
    config = dict(spec["training_config"])
    config["profiles"] = {name: Profile(**fields) for name, fields in config["profiles"].items()}
    # Explicit None: init arguments win over the environment and .env, which the
    # settings would otherwise read the keys from again.
    settings = Settings(**spec["settings"], **dict.fromkeys(SECRET_SETTINGS))
    return spec["req"], settings, TrainingConfig(**config), Profile(**spec["profile"])


def _child_env() -> dict[str, str]:
    """This process' environment minus the API keys (matched regardless of case, as the
    settings match them)."""
    return {name: value for name, value in os.environ.items()
            if name.upper() not in _SECRET_ENV}


class _StagingRegistry(Registry):
    """``run_training``'s registry in the child: it saves by staging only."""

    staged = False

    def save(self, name: str, model: ClassifierModel, metadata: dict,
             on_step: Callable[[str], None] = lambda _msg: None, *,
             overwrite: bool = False) -> None:
        self.stage(name, model, metadata, on_step, overwrite=overwrite)
        self.staged = True


def serve(spec: dict, emit: Callable[[dict], None], should_stop: Callable[[], bool]) -> int:
    """The child's work: run the job, report it through ``emit``. Returns the exit code."""
    from .training import run_training  # the heavy imports belong to the child only

    try:
        req, settings, training_cfg, profile = read_job(spec)
        registry = _StagingRegistry(settings.models_dir, settings.max_models_in_memory)
        result = run_training(req, settings, training_cfg, profile, registry,
                              on_progress=lambda **fields: emit({"progress": fields}),
                              should_stop=should_stop)
    except UserFacingError as exc:  # the job runner's contract: shown as it is
        emit({"failed": {"message": str(exc), "user_facing": True}})
        return 1
    except Exception:  # noqa: BLE001 - reported, logged, and never with its text
        logger.exception("Training in the child process failed")
        emit({"failed": {"message": "Training failed; see server logs for details.",
                         "user_facing": False}})
        return 1
    emit({"done": {"result": result, "staged": registry.staged}})
    return 0


def main() -> int:
    """Child entry point (``python -m app.train_worker``)."""
    # The protocol owns the real stdout. Anything else a library prints — at Python or at
    # C level — goes to stderr instead of corrupting a line the parent is parsing.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    first = sys.stdin.readline()
    if not first:
        return 3  # the parent went away before sending the job
    spec = json.loads(first)["job"]
    # Upper-cased like main.py does: `logging` refuses "info", and the child would die
    # before its first line over a spelling the server accepts.
    logging.basicConfig(level=str(spec["settings"].get("log_level", "INFO")).upper(),
                        format="%(asctime)s %(levelname)s [train-worker] %(name)s: %(message)s")
    # Here, not in serve(): the tests call serve() inside the test process, which must not
    # start counting itself twice — nor volunteer for the OOM killer.
    share_budget_with(spec["parent_pid"])
    _volunteer_for_the_oom_killer()
    stop = threading.Event()

    def listen() -> None:
        # Also the orphan guard, so nothing may end it early: a line that is no message
        # of ours is skipped, and whatever ends the loop ends the process.
        try:
            for line in sys.stdin:
                try:
                    if json.loads(line).get("stop"):
                        stop.set()
                except (ValueError, AttributeError):
                    continue
        finally:
            os._exit(3)  # stdin closed: the parent is gone or wants this run dead — now

    threading.Thread(target=listen, name="stop-listener", daemon=True).start()
    lock = threading.Lock()

    def emit(message: dict) -> None:
        with lock:  # progress can come from the sampler-side threads of the run
            protocol.write(json.dumps(message) + "\n")

    # This process' own id — on Windows the parent's Popen may only know a venv
    # launcher — so the status can read the training's live memory.
    emit({"progress": {"worker_pid": os.getpid()}})
    code = serve(spec, emit, stop.is_set)
    protocol.flush()
    return code


def _volunteer_for_the_oom_killer(path: Path = _OOM_SCORE_ADJ) -> None:
    """Make this process the one the kernel's OOM killer takes first.

    Without a hint it takes the LARGEST process, and with a big model cache that can be
    the API — which would end the run anyway, and the server with it. Raising one's own
    score needs no privilege; where the file does not exist (Windows, macOS) nothing
    happens. No help where a whole container is killed at once: Kubernetes 1.28+ on
    cgroup v2 sets memory.oom.group unless the kubelet's singleProcessOOMKill is on.
    """
    with contextlib.suppress(OSError):
        path.write_text("1000")


def _no_result_message(code: int) -> str:
    """What the job says about a child that ended without a result line."""
    message = f"The training process ended without a result (exit code {code})."
    if code in _KILLED_BY_SIGNAL_9:
        return (f"{message} An exit by signal 9 is almost always the out-of-memory killer: "
                "lower APIV3_TRAIN_MEMORY_MB or give the container more memory.")
    return f"{message} Its log is in the server log."


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "app.train_worker"]


def run_in_child(
    req: dict,
    settings: Settings,
    training_cfg: TrainingConfig,
    profile: Profile,
    registry: Registry,
    *,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
    kill_requested: Callable[[], bool] = lambda: False,
) -> dict:
    """``run_training`` in a child process, for the job runner: same arguments, same
    result, same exceptions — plus ``kill_requested``, which ends the child at once."""
    name = req["model_name"]
    # errors="replace": a byte that is not UTF-8 must cost one unreadable line, not the
    # reader thread and with it every line after.
    child = subprocess.Popen(_worker_command(), cwd=_APP_ROOT, env=_child_env(),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
    outcome: dict[str, Any] = {}
    try:
        # A child that died on start cannot take the job; its exit code says why below.
        _send(child.stdin, {"job": job_spec(req, settings, training_cfg, profile)})
        outcome = _relay(child, on_progress, should_stop, kill_requested)
    finally:
        _close(child.stdin)
        try:
            code = child.wait(timeout=_EXIT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
        # Nothing will publish what the child staged — also when the relay itself failed.
        # Only now: until the child is gone, it may still be writing there.
        if "done" not in outcome:
            registry.discard_staged(name)
    # Reaped: its id may name another process by now, and the publish below can wait on
    # the disk lock while the status still reads it.
    on_progress(worker_pid=None)
    if "done" in outcome:
        if outcome["done"]["staged"]:
            registry.publish(name, on_step=lambda detail: on_progress(phase_detail=detail))
        return outcome["done"]["result"]
    if "failed" in outcome:
        failed = outcome["failed"]
        if failed["user_facing"]:
            raise TrainingInputError(failed["message"])
        raise RuntimeError("the training process failed; its traceback is in the log")
    if kill_requested():
        return {}
    raise TrainingProcessError(_no_result_message(code))


def _relay(child: subprocess.Popen, on_progress: Callable[..., None],
           should_stop: Callable[[], bool], kill_requested: Callable[[], bool]) -> dict:
    """Forward the child's progress, pass a stop on, end it on a kill; return its outcome."""
    lines: queue.Queue[str | None] = queue.Queue()

    def pump(stream: IO[str]) -> None:
        try:
            for line in stream:
                lines.put(line)
        finally:
            lines.put(None)  # however reading ended, the relay must hear that it did

    threading.Thread(target=pump, args=(child.stdout,), name="child-stdout", daemon=True).start()
    stop_sent = False
    while True:
        if kill_requested():
            return {}
        if not stop_sent and should_stop():
            _send(child.stdin, {"stop": True})
            stop_sent = True
        try:
            line = lines.get(timeout=0.25)
        except queue.Empty:
            continue
        if line is None:
            return {}  # the child is gone without a word: its exit code tells the rest
        try:
            message = json.loads(line)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            # Half a line (the OOM killer does not wait for a newline), or output that is
            # no message of ours. What ends the run is end of output and the exit code.
            logger.warning("Training process sent a line that is no message; ignoring it.")
            continue
        if "progress" in message:
            on_progress(**message["progress"])
        elif "done" in message or "failed" in message:
            return message


def _send(stream: IO[str] | None, message: dict) -> None:
    if stream is None:
        return
    try:
        stream.write(json.dumps(message) + "\n")
        stream.flush()
    except (OSError, ValueError):  # the child already exited and closed its end
        pass


def _close(stream: IO[str] | None) -> None:
    try:
        if stream is not None:
            stream.close()
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
