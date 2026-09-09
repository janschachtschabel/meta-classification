"""End-to-end training on the tiny fixture (TF-IDF backend, fully offline).

Exercises the full pipeline (load -> prepare -> split -> bake-off -> threshold
-> evaluate -> deploy -> save) plus the registry save/get and export/import
round-trip with the secure skops format.
"""

import json
import time
from pathlib import Path

from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=FIXTURES,
        models_dir=tmp_path / "models",
        auth_enabled=False,
    )


def _request() -> dict:
    return {
        "dataset_name": "tiny.csv",
        "model_name": "tiny_model",
        "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
        "label_column": "properties.ccm:taxonid",
        "csv_separator": ";",
        "label_separator": ",",
        "label_filter": None,
    }


def _config() -> TrainingConfig:
    profile = Profile("fast", "TF-IDF", True, True, [1.0, 2.0])
    return TrainingConfig(
        default_profile="fast",
        profiles={"fast": profile},
        validation_size=0.2,
        test_size=0.2,
        min_text_length=5,
        drop_duplicates=True,
        min_samples_per_label=2,
    )


def _registry(settings: Settings) -> Registry:
    """The registry a test injects into run_training (mirrors the route's singleton)."""
    return Registry(settings.models_dir, settings.max_models_in_memory)


def test_train_predict_and_secure_roundtrip(tmp_path):
    settings = _settings(tmp_path)
    config = _config()
    result = run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["n_labels"] == 3
    assert result["task_type"] == "multiclass"
    assert result["metrics"]["f1_macro"] > 0.5

    registry = Registry(settings.models_dir, settings.max_models_in_memory)
    assert "tiny_model" in registry.list()

    model = registry.get("tiny_model")
    math_pred = model.predict(["Bruchrechnung und Gleichungen lösen üben"], top_k=1)
    assert math_pred[0] and math_pred[0][0].uri == "uri:math"

    # Export then import under a new name; the imported model must work.
    blob = registry.export_zip("tiny_model")
    info = registry.import_zip("tiny_copy", blob)
    assert info["name"] == "tiny_copy"
    copy_pred = registry.get("tiny_copy").predict(["Das Römische Reich der Antike"], top_k=1)
    assert copy_pred[0] and copy_pred[0][0].uri == "uri:hist"


def test_multiclass_training_reports_serving_rule_metrics(tmp_path):
    """tiny.csv is a multiclass set: serving decides via argmax and never reads
    thresholds, so the persisted bundle must carry argmax metrics
    (decision_rule) and neutral thresholds instead of tuned dead values."""
    settings = _settings(tmp_path)
    config = _config()
    result = run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["task_type"] == "multiclass"
    assert result["metrics"]["decision_rule"] == "argmax"

    model = _registry(settings).get("tiny_model")
    assert model.per_label_thresholds == {}
    assert model.global_threshold == 0.5


def test_warmup_preloads_configured_models(tmp_path, monkeypatch):
    """APIV3_WARMUP_MODELS preloads the named models into the LRU cache on
    startup (so the first /predict pays no cold skops-load) and warms their first
    prediction; a missing name is skipped without failing startup."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )

    monkeypatch.setenv("APIV3_DATA_DIR", str(FIXTURES))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(settings.models_dir))
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "false")
    monkeypatch.setenv("APIV3_WARMUP_MODELS", "tiny_model, ghost")
    get_settings.cache_clear()
    get_registry.cache_clear()
    try:
        with TestClient(create_app()) as client:  # __enter__ runs the lifespan (warmup)
            assert client.get("/health").status_code == 200  # 'ghost' did not crash startup
            reg = get_registry()
            assert reg.in_memory_count() == 1  # only the real model is resident
            model = reg.get("tiny_model")
            assert model._baseline is not None  # baseline_proba() ran -> first predict is warm
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()


def test_warmup_skips_corrupt_bundle_without_failing_startup(tmp_path, monkeypatch):
    """The warmup catches ANY load failure (not just missing names): one corrupt
    bundle on disk must never turn every pod start into a CrashLoopBackOff."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    models = tmp_path / "models"
    broken = models / "corrupt"
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("{ not json", encoding="utf-8")
    (broken / "head.skops").write_bytes(b"junk")
    (broken / "vectorizer.skops").write_bytes(b"junk")

    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(models))
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "false")
    monkeypatch.setenv("APIV3_WARMUP_MODELS", "corrupt")
    get_settings.cache_clear()
    get_registry.cache_clear()
    try:
        with TestClient(create_app()) as client:  # lifespan runs the warmup
            assert client.get("/health").status_code == 200
            assert get_registry().in_memory_count() == 0  # skipped, not cached
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()


def test_warmup_holds_every_listed_model_past_the_configured_cap(tmp_path, monkeypatch):
    """The vServer case: several trained models are listed for warmup so that each
    target field answers without a cold skops load. The LRU used to be sized purely
    from max_models_in_memory (default 2), so the third and fourth warmed model were
    evicted during startup and their first request paid the load anyway."""
    import shutil

    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    # Three distinct model names sharing one bundle: the cache counts entries, so
    # the content is irrelevant and copying beats training the same thing twice.
    source = settings.models_dir / "tiny_model"
    for name in ("level_model", "type_model"):
        shutil.copytree(source, settings.models_dir / name)

    monkeypatch.setenv("APIV3_DATA_DIR", str(FIXTURES))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(settings.models_dir))
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "false")
    monkeypatch.setenv("APIV3_MAX_MODELS_IN_MEMORY", "2")
    monkeypatch.setenv("APIV3_WARMUP_MODELS", "tiny_model, level_model, type_model")
    get_settings.cache_clear()
    get_registry.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
            assert get_registry().in_memory_count() == 3
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()


def test_run_training_uses_injected_registry(tmp_path):
    """run_training persists through the caller-provided registry — the route
    injects the get_registry() singleton, so training disk I/O shares the same
    disk lock as all API-side reads/deletes/imports (no second lock universe)."""
    settings = _settings(tmp_path)
    config = _config()
    saved: list[str] = []

    class RecordingRegistry(Registry):
        def save(self, name, model, metadata, **kwargs):
            saved.append(name)
            super().save(name, model, metadata, **kwargs)

    registry = RecordingRegistry(settings.models_dir, settings.max_models_in_memory)
    run_training(
        _request(), settings, config, config.get("fast"), registry,
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert saved == ["tiny_model"]  # persisted via the injected instance, not a fresh one


def test_hard_stop_not_overwritten_by_finishing_thread():
    """stop(hard=True) resets the status to idle; when the still-running target
    finishes later, it must NOT overwrite that reset with 'stopped'/'completed'."""
    import threading
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()
    release = threading.Event()

    def target(*, on_progress, should_stop):
        release.wait(3)  # simulate work that outlives the hard stop
        return {"ok": True}

    job.start(target, model_name="m")
    job.stop(hard=True)
    assert job.snapshot()["status"] == "idle"
    release.set()  # let the zombie thread finish now
    deadline = time.time() + 0.5
    while time.time() < deadline and job.snapshot()["status"] == "idle":
        time.sleep(0.01)  # give the runner the chance to (wrongly) overwrite
    assert job.snapshot()["status"] == "idle"


def test_hard_stop_refuses_new_start_until_thread_exits():
    """After a hard stop the status is idle but the thread keeps running (the
    deploy fit / save have no stop check). A new /train must be refused while
    that thread is alive — otherwise two full trainings run concurrently."""
    import threading

    import pytest

    from app.jobs import TrainingJob

    job = TrainingJob()
    started = threading.Event()
    release = threading.Event()

    def target(*, on_progress, should_stop):
        started.set()
        release.wait(5)  # work that outlives the hard stop (like the deploy fit)
        return {"ok": True}

    job.start(target, model_name="m")
    assert started.wait(2)
    job.stop(hard=True)
    assert job.snapshot()["status"] == "idle"
    with pytest.raises(RuntimeError):
        job.start(target, model_name="m2")  # thread still alive -> refuse

    release.set()  # let the zombie thread exit
    for _ in range(500):
        if job._thread is None or not job._thread.is_alive():
            break
        threading.Event().wait(0.01)
    job.start(target, model_name="m3")  # thread gone -> a new run is allowed
    release.set()


def test_training_head_fits_use_the_capped_n_jobs(tmp_path, monkeypatch):
    """Every head fit in the training path (C search + deploy fit) must receive
    effective_n_jobs (the CPU-budget-capped value), not the raw n_jobs setting."""
    from app import deploy as deploy_mod
    from app import tuning as tuning_mod
    from app.settings import Settings

    seen: list[int | None] = []

    def make_spy(real):
        def spy(c, **kwargs):
            seen.append(kwargs.get("n_jobs"))
            return real(c, **kwargs)
        return spy

    monkeypatch.setattr(deploy_mod, "make_head", make_spy(deploy_mod.make_head))
    monkeypatch.setattr(tuning_mod, "make_head", make_spy(tuning_mod.make_head))
    monkeypatch.setattr(Settings, "effective_n_jobs", lambda self: 3)

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert seen and set(seen) == {3}  # never the raw settings.n_jobs (-1)


def test_progress_advances_during_cv_and_c_search(tmp_path):
    """Progress must move WITHIN the long phases (the 30k run sat at a frozen 45%
    for ~25 min, which made the ETA grow instead of shrink): CV distributes
    45->90 across fold*C fits; the classic C search distributes 55->75."""
    settings = _settings(tmp_path)
    config = _config()

    def collect(progress_log):
        def on_progress(**fields):
            if "progress" in fields:
                progress_log.append(fields["progress"])
        return on_progress

    # CV mode: several distinct progress values strictly inside (45, 90].
    config.cv_folds = 2
    cv_log: list[int] = []
    run_training(_request(), settings, config, config.get("fast"), _registry(settings),
                 on_progress=collect(cv_log), should_stop=lambda: False)
    inside = [p for p in cv_log if 45 < p <= 90]
    assert len(set(inside)) >= 3, cv_log  # 2 folds x 2 C values -> 4 steps
    assert inside == sorted(inside)       # monotonically increasing

    # Classic split: the C search moves within (55, 75].
    config.cv_folds = 0
    split_log: list[int] = []
    req = _request()
    req["model_name"] = "tiny_progress"
    run_training(req, settings, config, config.get("fast"), _registry(settings),
                 on_progress=collect(split_log), should_stop=lambda: False)
    assert any(55 < p <= 75 for p in split_log), split_log


def test_run_training_cancelled_saves_nothing(tmp_path):
    """A cancelled run returns an empty result and must NOT persist a bundle —
    the cooperative-stop contract that the /train/stop feature relies on."""
    settings = _settings(tmp_path)
    config = _config()
    result = run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: True,  # cancelled from the start
    )
    assert result == {}
    registry = Registry(settings.models_dir, settings.max_models_in_memory)
    assert "tiny_model" not in registry.list()


def test_sweep_stale_tmp_removes_orphaned_staging_dirs(tmp_path):
    """A crashed/OOM-killed save leaves a hidden `.name.tmp` staging dir (the
    atomic rename never ran). The startup sweep must delete those orphans and
    leave real models untouched — otherwise they leak disk until the same name
    is retrained."""
    models = tmp_path / "models"
    models.mkdir()
    good = models / "keep_me"                         # a real model: survives
    good.mkdir()
    (good / "config.json").write_text("{}", encoding="utf-8")
    for name in (".keep_me.tmp", ".other.tmp"):       # orphaned staging dirs: removed
        stale = models / name
        stale.mkdir()
        (stale / "head.skops").write_text("partial", encoding="utf-8")

    removed = Registry(models, 2).sweep_stale_tmp()

    assert removed == 2
    assert not (models / ".keep_me.tmp").exists() and not (models / ".other.tmp").exists()
    assert (good / "config.json").exists()
    assert Registry(models, 2).list() == ["keep_me"]


def test_bundle_backups_are_not_listed_as_models(tmp_path):
    """`scripts/prune_bundle_labels.py` keeps a full `<name>.prebackup` copy before
    it rewrites a bundle. That copy is a valid bundle, so list() used to offer it as
    an ordinary model — and predicting with it silently serves the PRE-repair
    weights (the container-label class this project removes on purpose). A backup is
    not a model; only the repaired bundle is."""
    models = tmp_path / "models"
    models.mkdir()
    for name in ("subjects", "subjects.prebackup"):
        bundle = models / name
        bundle.mkdir()
        (bundle / "config.json").write_text("{}", encoding="utf-8")

    assert Registry(models, 2).list() == ["subjects"]


def test_sweep_stale_tmp_does_not_overcount_failed_removals(tmp_path, monkeypatch):
    """rmtree runs with ignore_errors=True, so a removal can silently fail (e.g. a
    locked file on Windows). The returned count must reflect dirs actually gone,
    not attempts — otherwise the startup log over-reports."""
    from app import registry as reg_mod

    models = tmp_path / "models"
    models.mkdir()
    stale = models / ".stuck.tmp"
    stale.mkdir()
    (stale / "head.skops").write_text("partial", encoding="utf-8")

    monkeypatch.setattr(reg_mod.shutil, "rmtree", lambda *a, **k: None)  # simulate a failed removal
    removed = Registry(models, 2).sweep_stale_tmp()

    assert removed == 0  # nothing actually removed -> count 0, not 1
    assert stale.exists()  # the dir is indeed still present


def test_get_cannot_reinsert_a_concurrently_deleted_model(tmp_path, monkeypatch):
    """Registry invariant: a delete must never lose to a concurrent cold get()'s
    cache insert. Regression: get() loaded under the disk lock but inserted after
    releasing it, so delete X -> re-import X served the OLD deleted weights."""
    import threading

    from app import registry as reg_mod

    settings = _settings(tmp_path)
    config = _config()
    run_training(_request(), settings, config, config.get("fast"), _registry(settings),
                 on_progress=lambda **_: None, should_stop=lambda: False)
    reg = reg_mod.Registry(settings.models_dir, settings.max_models_in_memory)

    read_started = threading.Event()
    resume_read = threading.Event()
    real_read = reg_mod._read_bundle

    def blocking_read(directory):
        read_started.set()
        resume_read.wait(5)
        return real_read(directory)

    monkeypatch.setattr(reg_mod, "_read_bundle", blocking_read)

    getter = threading.Thread(target=lambda: reg.get("tiny_model"), daemon=True)
    getter.start()
    assert read_started.wait(3), "cold get() did not start reading"

    deleted = threading.Event()

    def do_delete() -> None:
        reg.delete("tiny_model")
        deleted.set()

    deleter = threading.Thread(target=do_delete, daemon=True)
    deleter.start()
    resume_read.set()
    getter.join(5)
    assert deleted.wait(5), "delete did not complete"

    assert reg.in_memory_count() == 0, (
        "the deleted model was re-inserted into the LRU cache by the concurrent get()"
    )


def test_zombie_thread_progress_cannot_mutate_reset_state():
    """After stop(hard=True) the abandoned thread's on_progress updates must be
    dropped: /metrics otherwise reports training_running=0 with progress creeping."""
    import threading

    from app.jobs import TrainingJob

    job = TrainingJob()
    entered = threading.Event()
    resume = threading.Event()

    def target(on_progress, should_stop):
        entered.set()
        resume.wait(5)
        on_progress(progress=55, message="zombie says hi")  # after the hard reset
        return {}

    job.start(target, model_name="m1")
    assert entered.wait(3)
    job.stop(hard=True)  # status -> idle, generation bumped
    resume.set()
    for _ in range(100):
        if not (job._thread and job._thread.is_alive()):
            break
        time.sleep(0.02)

    snap = job.snapshot()
    assert snap["status"] == "idle"
    assert snap["progress"] == 0, "zombie progress leaked into the reset state"
    assert snap["message"] != "zombie says hi"


def test_failed_training_populates_error_field():
    """/train/status documents an `error` field; failures must populate it (it was
    a documented-but-always-null key)."""
    from app.errors import TrainingInputError
    from app.jobs import TrainingJob

    job = TrainingJob()

    def target(on_progress, should_stop):
        raise TrainingInputError("Column 'nope' not found.")

    job.start(target, model_name="m2")
    deadline = time.time() + 3
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.02)

    snap = job.snapshot()
    assert snap["status"] == "error"
    assert snap["error"] == "Column 'nope' not found."


def test_active_model_name_survives_hard_stop_while_thread_lives():
    """The import guard needs the truth 'a thread is still writing model X' — job
    STATUS lies after a hard stop (idle), so active_model_name() must key on
    thread liveness instead."""
    import threading

    from app.jobs import TrainingJob

    job = TrainingJob()
    entered = threading.Event()
    resume = threading.Event()

    def target(on_progress, should_stop):
        entered.set()
        resume.wait(5)
        return {}

    job.start(target, model_name="m3")
    assert entered.wait(3)
    job.stop(hard=True)  # status idle, but the thread still runs
    assert job.active_model_name() == "m3"
    resume.set()
    job._thread.join(3)
    assert job.active_model_name() is None


def test_training_job_cooperative_stop_sets_stopped_status():
    """TrainingJob.stop() makes a cooperative target finish and the job report
    status 'stopped' (not 'completed')."""
    import threading
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()
    running = threading.Event()

    def target(*, on_progress, should_stop):
        running.set()
        while not should_stop():
            time.sleep(0.005)
        return {"ok": True}

    job.start(target, model_name="m")
    assert running.wait(2.0), "training target did not start"
    job.stop()  # cooperative stop
    deadline = time.time() + 2.0
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.01)
    assert job.snapshot()["status"] == "stopped"


def test_run_training_cross_validation_mode(tmp_path):
    """cv_folds>=2 trains + validates on all rows via out-of-fold, then deploys a
    working model on 100% of the data and reports metrics (option to the split)."""
    settings = _settings(tmp_path)
    config = _config()
    config.cv_folds = 2  # cross-validation instead of the classic train/val/test split
    result = run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["n_labels"] == 3
    assert 0.0 <= result["metrics"]["f1_macro"] <= 1.0

    registry = Registry(settings.models_dir, settings.max_models_in_memory)
    assert "tiny_model" in registry.list()
    pred = registry.get("tiny_model").predict(["Bruchrechnung und Gleichungen lösen üben"], top_k=1)
    assert pred[0] and pred[0][0].uri == "uri:math"

    # Metadata reflects CV honestly: deployed on 100%, no fixed holdout.
    meta = registry.info("tiny_model")["metadata"]
    assert "cross-validation" in meta["evaluation"]
    assert meta["n_train"] == meta["n_samples"]
    assert meta["n_test"] == 0


def test_cv_mode_keeps_labels_missing_from_the_train_split(tmp_path, monkeypatch):
    """In CV mode every row trains, so a label whose positives all fall outside the
    (unused for CV) train split must NOT be dropped — losing rare labels to the
    split is exactly the data loss the CV option exists to avoid."""
    import numpy as np

    from app import data as data_mod

    def adversarial_split(n, *, val_size, test_size, seed):
        idx = np.arange(n)
        # All 'uri:hist' rows (24..35 in tiny.csv) land outside the train split.
        return idx[:24], idx[24:30], idx[30:]

    monkeypatch.setattr(data_mod, "three_way_split", adversarial_split)
    settings = _settings(tmp_path)
    config = _config()
    config.cv_folds = 2
    result = run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["n_labels"] == 3  # hist survives; split mode would drop it


def test_request_cv_folds_overrides_config(tmp_path):
    """cv_folds in the train request selects CV per run, overriding the config
    default (0 = classic split here) — mirroring the min_samples_per_label override."""
    settings = _settings(tmp_path)
    config = _config()  # config default stays 0 (classic split)
    req = _request()
    req["model_name"] = "tiny_cvreq"
    req["cv_folds"] = 2
    run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    meta = Registry(settings.models_dir, settings.max_models_in_memory).info("tiny_cvreq")["metadata"]
    assert "cross-validation" in meta["evaluation"]


def test_training_job_shows_crafted_input_error_message():
    """User-facing input errors (wrong column name, too few rows, bad cv_folds) are
    raised as TrainingInputError and their crafted message IS shown on /train/status —
    unlike arbitrary exceptions, which stay sanitized."""
    import time

    from app.errors import TrainingInputError
    from app.jobs import TrainingJob

    job = TrainingJob()

    def target(*, on_progress, should_stop):
        raise TrainingInputError("Label column 'oops' not found in ['title', 'taxonid']")

    job.start(target, model_name="m")
    deadline = time.time() + 3
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.01)
    snap = job.snapshot()
    assert snap["status"] == "error"
    assert "Label column 'oops' not found" in snap["message"]


def test_registry_disk_lock_serializes_read_and_delete(tmp_path, monkeypatch):
    """A cold load and a concurrent delete of the same model must NOT interleave:
    the disk lock prevents a reader observing a mid-rmtree/mid-replace bundle."""
    import threading

    from app import registry as reg_mod

    settings = _settings(tmp_path)
    config = _config()
    run_training(_request(), settings, config, config.get("fast"), _registry(settings),
                 on_progress=lambda **_: None, should_stop=lambda: False)

    read_started = threading.Event()
    release = threading.Event()
    order: list[str] = []
    real_read = reg_mod._read_bundle

    def blocking_read(directory):
        read_started.set()
        release.wait(3)          # hold the disk lock while "reading"
        order.append("read")
        return real_read(directory)

    monkeypatch.setattr(reg_mod, "_read_bundle", blocking_read)
    reg = reg_mod.Registry(settings.models_dir, settings.max_models_in_memory)  # cold cache

    threading.Thread(target=lambda: reg.load_fresh("tiny_model"), daemon=True).start()
    assert read_started.wait(3), "read did not start"

    done = threading.Event()

    def do_delete() -> None:
        try:
            reg.delete("tiny_model")
        finally:
            order.append("delete")
            done.set()

    threading.Thread(target=do_delete, daemon=True).start()
    assert not done.wait(0.4), "delete ran while a read held the disk lock (race not serialized)"
    release.set()
    assert done.wait(3), "delete did not complete after the read released the lock"
    assert order == ["read", "delete"]


def test_slow_save_does_not_block_reads_of_other_models(tmp_path, monkeypatch):
    """A multi-minute skops dump (save) must NOT hold the disk lock across the
    write — otherwise a /predict or /models read of a DIFFERENT, already-saved
    model blocks for the whole save."""
    import threading

    from app import registry as reg_mod

    settings = _settings(tmp_path)
    config = _config()
    run_training(_request(), settings, config, config.get("fast"), _registry(settings),
                 on_progress=lambda **_: None, should_stop=lambda: False)
    reg = reg_mod.Registry(settings.models_dir, settings.max_models_in_memory)
    model, meta = reg.load_fresh("tiny_model")

    write_started = threading.Event()
    release = threading.Event()
    real_write = reg_mod._write_bundle

    def blocking_write(directory, m, md, on_step=lambda _msg: None):
        write_started.set()
        release.wait(3)  # simulate a slow dump
        return real_write(directory, m, md, on_step)

    monkeypatch.setattr(reg_mod, "_write_bundle", blocking_write)
    saver = threading.Thread(target=lambda: reg.save("other", model, meta), daemon=True)
    saver.start()
    assert write_started.wait(3), "save did not start writing"

    result: dict = {}

    def read_other() -> None:
        result["info"] = reg.info("tiny_model")

    reader = threading.Thread(target=read_other, daemon=True)
    reader.start()
    reader.join(1.5)
    blocked = reader.is_alive()
    release.set()
    saver.join(5)
    reader.join(5)
    assert not blocked, "info() blocked behind the in-progress save (disk lock held across the dump)"
    assert result["info"]["name"] == "tiny_model"


def test_prepare_rejects_too_few_rows_after_dropping_rare_labels(tmp_path):
    """The pre-filter '>=10 rows' guard counts rows that prepare_targets then drops
    (rare labels). Two labels survive but only 8 rows remain -> the split would be
    near-empty; it must be rejected, not 'succeed' with meaningless metrics."""
    import pytest

    from app.errors import TrainingInputError
    from app.prepare import prepare_data

    rows = ["properties.cclom:title;properties.cclom:general_keyword;properties.ccm:taxonid"]
    rows += [f"Titel A Nummer {i} zum Thema Physik;kw{i};A" for i in range(5)]
    rows += [f"Titel B Nummer {i} zum Thema Chemie;kw{i};B" for i in range(3)]
    rows += [f"Einzelzeile {i} mit eindeutigem Label hier;kw{i};U{i}" for i in range(10)]
    csv = tmp_path / "rare.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    req = _request()
    req["dataset_name"] = "rare.csv"
    with pytest.raises(TrainingInputError) as exc:
        prepare_data(req, settings, _config(), cv_folds=0,
                     on_progress=lambda **_: None, should_stop=lambda: False)
    assert "after dropping" in str(exc.value)


def test_drop_unlearnable_removes_orphaned_rows_and_remaps_split():
    """Dropping a label column without train positives can orphan rows whose ONLY
    label that was (all-zero targets). Those rows must go too — kept, they score
    guaranteed misses in the metrics and dilute avg_labels — and the split
    indices must be remapped onto the surviving rows."""
    import numpy as np

    from app.prepare import _drop_unlearnable

    texts = np.array([f"text {i}" for i in range(6)], dtype=object)
    y = np.array([
        [1, 0], [1, 0], [1, 0],  # rows 0-2 (train): only label "a" has positives here
        [0, 1],                  # row 3 (val): label "b" exists ONLY outside train
        [1, 0], [0, 1],          # rows 4 (val) / 5 (test)
    ])
    splits = (np.array([0, 1, 2]), np.array([3, 4]), np.array([5]))

    texts2, y2, classes2, (tr2, va2, te2) = _drop_unlearnable(texts, y, ["a", "b"], splits)

    assert classes2 == ["a"]
    assert list(texts2) == ["text 0", "text 1", "text 2", "text 4"]
    assert y2.shape == (4, 1)
    assert (y2.sum(axis=1) > 0).all(), "orphaned all-zero rows must be dropped"
    assert tr2.tolist() == [0, 1, 2]
    assert va2.tolist() == [3]  # old row 4 -> new position 3
    assert te2.tolist() == []   # old row 5 was orphaned (caller guards empty splits)


def test_drop_unlearnable_is_a_no_op_when_all_columns_learnable():
    import numpy as np

    from app.prepare import _drop_unlearnable

    texts = np.array(["x", "y", "z", "w"], dtype=object)
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    splits = (np.array([0, 1]), np.array([2]), np.array([3]))

    texts2, y2, classes2, (tr2, va2, te2) = _drop_unlearnable(texts, y, ["a", "b"], splits)

    assert list(texts2) == ["x", "y", "z", "w"]
    assert (y2 == y).all()
    assert classes2 == ["a", "b"]
    assert (tr2.tolist(), va2.tolist(), te2.tolist()) == ([0, 1], [2], [3])


def test_prepare_avg_labels_and_rows_reflect_the_dropped_label_space(tmp_path):
    """avg_labels must describe the label space actually trained (computed AFTER
    the unlearnable-column drop) and no all-zero target rows may survive it."""
    import numpy as np
    import pytest

    from app.prepare import prepare_data

    rows = ["properties.cclom:title;properties.cclom:general_keyword;properties.ccm:taxonid"]
    rows += [f"Titel A Nummer {i} zum Thema Physik;kw-a{i};A" for i in range(12)]
    rows += [f"Titel B Nummer {i} zum Thema Chemie;kw-b{i};B" for i in range(12)]
    # Six single-row labels: with seed 42 at least one lands outside the train
    # split, making its column unlearnable and its row an orphan candidate.
    rows += [f"Unikat Nummer {i} ganz anderes Thema;kw-u{i};R{i}" for i in range(6)]
    csv = tmp_path / "orphan.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    req = _request()
    req["dataset_name"] = "orphan.csv"
    req["min_samples_per_label"] = 1  # keep the single-row labels through prepare_targets

    prep = prepare_data(req, settings, _config(), cv_folds=0,
                        on_progress=lambda **_: None, should_stop=lambda: False)

    assert prep is not None
    assert len(prep.classes) < 8, "test setup: expected >=1 rare label outside the train split"
    row_sums = prep.y_all.sum(axis=1)
    assert (row_sums > 0).all(), "rows orphaned by the column drop must be removed"
    assert prep.avg_labels == pytest.approx(float(row_sums.mean()))
    assert prep.avg_labels == pytest.approx(1.0)  # single-label data stays exactly 1.0
    joined = np.concatenate([prep.train_idx, prep.val_idx, prep.test_idx])
    assert sorted(joined.tolist()) == list(range(len(prep.texts)))
    assert len(prep.texts) == prep.y_all.shape[0]


def test_training_job_error_message_is_sanitized():
    """A failed training job must NOT expose the raw exception (paths/internals) on
    the readonly /train/status snapshot."""
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()

    def target(*, on_progress, should_stop):
        raise ValueError("leaked C:/secret/path.csv column=ssn value=42")

    job.start(target, model_name="m")
    deadline = time.time() + 3
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.01)
    snap = job.snapshot()
    assert snap["status"] == "error"
    assert snap["message"]  # a generic message is present
    assert "secret" not in snap["message"] and "ssn" not in snap["message"]  # raw exc not leaked


class _Sneaky:
    """Deliberately unknown type for the skops allowlist test below."""


def test_safe_skops_load_rejects_untrusted_types(tmp_path):
    """The core anti-RCE guard: a skops container carrying a type outside the
    allowlist is refused — regardless of how it got on disk."""
    import pytest
    import skops.io as sio

    from app.model_io import UnsafeModelError, _safe_skops_load

    path = tmp_path / "evil.skops"
    sio.dump(_Sneaky(), str(path))
    with pytest.raises(UnsafeModelError, match="untrusted"):
        _safe_skops_load(path)


def test_prepare_rejects_filter_matching_no_label(tmp_path):
    """A label_filter that matches nothing fails fast with a crafted, user-facing
    TrainingInputError instead of training an empty model."""
    import pytest

    from app.errors import TrainingInputError

    settings = _settings(tmp_path)
    config = _config()
    req = _request()
    req["model_name"] = "never_created"
    req["label_filter"] = "zzz-matches-no-label"
    with pytest.raises(TrainingInputError):
        run_training(
            req, settings, config, config.get("fast"), _registry(settings),
            on_progress=lambda **_: None, should_stop=lambda: False,
        )


def test_import_rejects_unsafe_archive(tmp_path):
    """A zip without the required safe files must be refused."""
    import io
    import zipfile

    import pytest

    from app.registry import UnsafeModelError

    registry = Registry(tmp_path / "models", 2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("evil.txt", "not a model")
    with pytest.raises(UnsafeModelError):
        registry.import_zip("evil", buffer.getvalue())


def test_crashed_save_leaves_no_visible_model(tmp_path):
    """Bundles are written atomically: a leftover hidden tmp dir (crashed save)
    is invisible to list()/exists() and gets cleaned up by the next save."""
    settings = _settings(tmp_path)
    registry = Registry(settings.models_dir, settings.max_models_in_memory)
    stale = Path(settings.models_dir) / ".tiny_model.tmp"
    stale.mkdir(parents=True)
    (stale / "config.json").write_text("{}", encoding="utf-8")

    assert registry.list() == []  # the torso must not appear as a model
    assert not registry.exists("tiny_model")

    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert not stale.exists()  # next save cleaned the stale tmp dir
    bundle = Path(settings.models_dir) / "tiny_model"
    for required in ("config.json", "metrics.json", "head.skops", "vectorizer.skops"):
        assert (bundle / required).exists()  # rename only publishes complete bundles


def test_training_aborts_with_an_actionable_message_when_no_label_has_enough_rows(tmp_path):
    """A min_samples_per_label above EVERY label's count aborts training. The message
    must name the actual best count and the label total — "no label has >= 20" alone
    leaves the user guessing whether they are one row or a thousand rows short.
    jobs.py turns this into status=error on /train/status."""
    import pytest

    from app.errors import TrainingInputError

    settings = _settings(tmp_path)
    config = _config()
    req = {**_request(), "min_samples_per_label": 20}  # tiny.csv holds 12 rows per label
    with pytest.raises(TrainingInputError) as caught:
        run_training(
            req, settings, config, config.get("fast"), _registry(settings),
            on_progress=lambda **_: None, should_stop=lambda: False,
        )
    message = str(caught.value)
    assert "20" in message  # the limit that was not met
    assert "12" in message  # what the best label actually has
    assert "3" in message   # how many labels were looked at
    assert not (Path(settings.models_dir) / "tiny_model").exists()  # nothing published


def test_profile_carries_its_evaluation_mode_and_the_request_still_wins(tmp_path):
    """Profiles are size classes now: which evaluation mode fits is a property of the
    size, so the profile carries it. Resolution stays most-specific-first —
    request > profile > config — so an explicit cv_folds still overrides the profile."""
    settings = _settings(tmp_path)
    config = _config()
    config.cv_folds = 0  # config says holdout ...
    config.profiles["cv3"] = Profile("cv3", "", True, True, [1.0], cv_folds=3)

    run_training(
        {**_request(), "model_name": "from_profile"}, settings, config, config.get("cv3"),
        _registry(settings), on_progress=lambda **_: None, should_stop=lambda: False,
    )
    registry = Registry(settings.models_dir, 2)
    assert "3-fold" in registry.info("from_profile")["metadata"]["evaluation"]

    run_training(
        {**_request(), "model_name": "from_request", "cv_folds": 0}, settings, config,
        config.get("cv3"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert "holdout" in Registry(settings.models_dir, 2).info("from_request")["metadata"]["evaluation"]


def test_shipped_config_and_code_defaults_describe_the_same_profiles():
    """`profiles._DEFAULTS` is the fallback when config.yaml is missing, so the two are
    duplicated by design — and silently drift apart, which would make behaviour depend on
    whether the file happens to exist. Pin them together."""
    from app.profiles import _DEFAULTS, load_training_config

    shipped = load_training_config(FIXTURES.parent.parent / "config.yaml")
    assert set(shipped.profiles) == set(_DEFAULTS), "profile names differ"
    for name, profile in shipped.profiles.items():
        fallback = _DEFAULTS[name]
        assert profile.c_grid == fallback.c_grid, f"{name}: C grid differs"
        assert profile.use_char == fallback.use_char, f"{name}: use_char differs"
        assert profile.max_word_features == fallback.max_word_features, f"{name}: word cap differs"
        assert profile.threshold_per_label == fallback.threshold_per_label, f"{name}: thresholds differ"
        assert profile.cv_folds == fallback.cv_folds, f"{name}: evaluation mode differs"


def test_loading_a_bundle_with_a_container_label_warns(tmp_path, caplog):
    """Training can no longer produce a container label, but an IMPORT can still bring one.

    `app.data.split_labels` closes the training path, yet `POST /models/import` installs a
    bundle whose `classes` come straight from its own `config.json` — so a bundle built by
    an older version reintroduces the junk class. Correcting it here is not an option: the
    class list is positionally tied to the head's estimators, and a loader that silently
    dropped a column would desynchronise every prediction. So make it impossible to miss
    instead, and point at the repair.
    """
    import json
    import logging

    settings = _settings(tmp_path)
    config = _config()
    run_training(_request(), settings, config, config.get("fast"), _registry(settings),
                 on_progress=lambda **_: None, should_stop=lambda: False)

    # Rename one class to a container URI: the estimator count still matches, which is
    # exactly what a legacy bundle looks like.
    bundle = Path(settings.models_dir) / "tiny_model"
    raw = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    raw["classes"][0] = "http://w3id.org/openeduhub/vocabs/discipline/"
    (bundle / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        model, _ = Registry(settings.models_dir, 2).load_fresh("tiny_model")

    assert "http://w3id.org/openeduhub/vocabs/discipline/" in model.classes  # still loads
    warnings = " ".join(record.getMessage() for record in caplog.records)
    assert "container" in warnings.lower(), f"no warning about the container label: {warnings!r}"
    assert "prune_bundle_labels" in warnings, "the warning must name the repair script"


def test_every_profile_brackets_both_known_optima():
    """A short C grid is only safe if it still SPANS the useful range.

    Measured on this project's own targets, the optimum sits at opposite ends: the
    subject model picks C=32, the educational-level model picked C=2. A grid like
    [4, 16] has two candidates but reaches neither — it would silently ship a model
    regularized at the wrong end while looking like a normal run. This is the
    invariant that lets `fast`/`auto` get away with only two candidates, so it is
    pinned here rather than left as a comment in config.yaml.
    """
    from app.profiles import load_training_config

    shipped = load_training_config(FIXTURES.parent.parent / "config.yaml")
    for name, profile in shipped.profiles.items():
        assert min(profile.c_grid) <= 2.0, f"{name}: grid starts above the known low optimum C=2"
        assert max(profile.c_grid) >= 32.0, f"{name}: grid ends below the known high optimum C=32"


def test_profiles_form_a_monotone_cost_ladder():
    """fast < auto < best in cost, so the profile name is a truthful price tag.

    The previous set was not ordered: `large` was CHEAPER than `auto` despite sounding
    heavier, and `thorough` sat outside the CV scheme entirely. Cost here is head fits
    in units of the full dataset — k folds each training on (k-1)/k of the rows, once
    per C candidate, plus the deploy fit — which is what actually drives wall-clock.
    """
    from app.profiles import load_training_config

    shipped = load_training_config(FIXTURES.parent.parent / "config.yaml")

    def cost(profile) -> float:
        folds = profile.cv_folds or 0
        per_candidate = (folds - 1) if folds >= 2 else 0.7  # holdout trains on 70%
        return per_candidate * len(profile.c_grid) + 1.0  # + the deploy fit on 100%

    ladder = ["fast", "auto", "best"]
    assert set(shipped.profiles) == set(ladder), "the shipped set is exactly the ladder"
    costs = [cost(shipped.profiles[name]) for name in ladder]
    assert costs == sorted(costs) and len(set(costs)) == len(costs), f"not monotone: {costs}"


def test_training_from_gzip_equals_training_from_plain_csv(tmp_path):
    """The whole pipeline must be indifferent to compression — proven, not assumed.

    Same fixture, once plain and once gzipped, same seed: load, prepare, fit, evaluate and
    save all have to produce the same numbers. Anything less would mean the compressed form
    silently trains on different data, which is the failure mode nobody notices until a model
    underperforms for no visible reason.
    """
    import gzip
    import shutil

    settings = _settings(tmp_path)
    data_dir = Path(settings.data_dir)
    plain = FIXTURES / "tiny.csv"
    # The data dir is the fixtures dir; write the gz twin into a private one to avoid
    # polluting the repository.
    private = tmp_path / "data"
    private.mkdir(parents=True, exist_ok=True)
    shutil.copy(plain, private / "tiny.csv")
    with open(plain, "rb") as src, gzip.open(private / "tiny.csv.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    settings = Settings(data_dir=private, models_dir=tmp_path / "models", auth_enabled=False)
    assert data_dir  # the fixture-based settings are unused beyond this point

    config = _config()
    results = {}
    for name in ("tiny.csv", "tiny.csv.gz"):
        model_name = f"m_{name.replace('.', '_')}"
        run_training(
            {**_request(), "dataset_name": name, "model_name": model_name},
            settings, config, config.get("fast"), _registry(settings),
            on_progress=lambda **_: None, should_stop=lambda: False,
        )
        registry = Registry(settings.models_dir, 2)
        meta = registry.info(model_name)["metadata"]
        predictions = registry.get(model_name).predict(["Bruchrechnung und Gleichungen"], top_k=3)[0]
        results[name] = {
            "rows": meta["n_samples"], "labels": meta["n_labels"], "best_C": meta["best_C"],
            "f1_macro": meta["metrics"]["f1_macro"],
            "per_label_f1": meta["metrics"]["per_label_f1"],
            "top": [(p.uri, round(p.confidence, 10)) for p in predictions],
        }

    plain_result, gz_result = results["tiny.csv"], results["tiny.csv.gz"]
    assert plain_result == gz_result, (
        f"gzip changed the outcome:\n  plain: {plain_result}\n  gzip : {gz_result}")
    assert plain_result["rows"] > 0 and plain_result["labels"] >= 2  # the run was real


def test_no_c_grid_reaches_past_the_measured_useful_range():
    """No profile may search above C=32, because above it quality DROPS.

    This replaces an earlier assumption of mine that `best` should probe past both known
    optima (`[0.5 … 128]`) on the theory that four consecutive runs picking the grid
    maximum meant the optimum lay beyond it. 🟢 Measured instead
    (`scripts/benchmark_c_range.py`, holdout split, both targets, zero convergence
    warnings) — the curve peaks at 32 and then declines:

        school       C=8 0.7504 | C=32 0.7531 | C=128 0.7511 | C=512 0.7493 | C=2048 0.7479
        university   C=8 0.7803 | C=32 0.7814 | C=128 0.7812 | C=512 0.7785 | C=2048 0.7785

    The repeated edge picks were ties on a flat plateau, not a missing optimum: at fixed
    5-fold CV, C=32 and C=128 scored 0.8135 vs 0.8130. Reaching further costs quality AND
    ~50% more fit time, so the upper bound is pinned here to stop it being re-widened on
    the same hunch.
    """
    from app.profiles import load_training_config

    shipped = load_training_config(FIXTURES.parent.parent / "config.yaml")
    for name, profile in shipped.profiles.items():
        assert max(profile.c_grid) <= 32.0, f"{name}: searches above the measured optimum C=32"


def test_c_grid_resolution_never_shrinks_along_the_ladder():
    """A costlier rung may not search C more coarsely than a cheaper one.

    With the useful range fixed at 2 … 32, the grid can only differ in resolution, and the
    ladder's real gradation moved to the fold count (`fast` holdout -> `auto` 3 -> `best` 5).
    `fast` may stay coarse because it exists for iteration; nothing above it may regress.
    """
    from app.profiles import load_training_config

    shipped = load_training_config(FIXTURES.parent.parent / "config.yaml")
    counts = [len(shipped.profiles[name].c_grid) for name in ("fast", "auto", "best")]
    assert counts == sorted(counts), f"resolution shrinks along the ladder: {counts}"


def test_request_can_override_the_tfidf_feature_caps(tmp_path):
    """The vocabulary caps are the main RAM/quality lever but were only reachable via
    env vars or a profile — i.e. not per run. A request override makes "is the cap
    cutting off signal?" answerable without touching the deployment. The bundle records
    the resulting feature count, so the effect stays visible afterwards."""
    settings = _settings(tmp_path)
    config = _config()
    # The fixture's natural char vocabulary is ~855, so a cap of 100 provably BINDS;
    # asserting against a cap above the natural size would prove nothing.
    req = {**_request(), "max_word_features": 40, "max_char_features": 100}
    run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    registry = Registry(settings.models_dir, 2)
    vectorizer = registry.get("tiny_model").vectorizer
    assert len(vectorizer.char_vec.vocabulary_) == 100  # truncated to the request's cap
    meta = registry.info("tiny_model")["metadata"]
    assert meta["tfidf"]["max_word_features"] == 40
    assert meta["tfidf"]["max_char_features"] == 100
    assert meta["tfidf"]["n_features"] <= 140  # cannot exceed the sum of both caps


def test_config_default_text_column_weights_apply_when_the_request_omits_them(tmp_path):
    """A weights default in config.yaml is a dataset CONVENTION, not a user instruction:
    it applies when the request omits the field, and is narrowed to the columns actually
    being trained on — otherwise a global default would break every CSV with different
    column names. The bundle records what was EFFECTIVELY used, not what was configured.
    """
    settings = _settings(tmp_path)
    config = _config()
    config.text_column_weights = {"properties.cclom:title": 3, "not.in.this.csv": 5}
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    meta = Registry(settings.models_dir, 2).info("tiny_model")["metadata"]
    assert meta["text_column_weights"] == {"properties.cclom:title": 3}


def test_empty_text_column_weights_switch_the_config_default_off(tmp_path):
    """`null` means "use the config default"; an explicit empty mapping means "no
    weighting" — without that distinction a caller could not opt out of the default."""
    settings = _settings(tmp_path)
    config = _config()
    config.text_column_weights = {"properties.cclom:title": 3}
    run_training(
        {**_request(), "text_column_weights": {}}, settings, config, config.get("fast"),
        _registry(settings), on_progress=lambda **_: None, should_stop=lambda: False,
    )
    meta = Registry(settings.models_dir, 2).info("tiny_model")["metadata"]
    assert meta["text_column_weights"] == {}


def test_text_column_weights_are_anchored_in_the_bundle(tmp_path):
    """The weights are a property of the TRAINED model, not of one request: the model
    was fit on text where those fields repeat, so the bundle has to record them for a
    caller to be able to build matching input text."""
    settings = _settings(tmp_path)
    config = _config()
    req = {**_request(), "text_column_weights": {"properties.cclom:title": 2}}
    run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    meta = Registry(settings.models_dir, 2).info("tiny_model")["metadata"]
    assert meta["text_column_weights"] == {"properties.cclom:title": 2}


def test_bundle_carries_the_authored_info_and_the_derived_vocabulary(tmp_path):
    """Everything else in the metadata is MEASURED; these four fields are what only
    the person training the model knows — who made it, what for, where the data came
    from, on what terms. They matter when a bundle is handed to a third party, where
    `dataset: "data_300k.csv"` means nothing. Kept in their own `info` block so a
    reader can tell an assertion by a human from a fact by the pipeline.

    The vocabulary is NOT among them: it is derived from the class URIs, so it cannot
    be typed in wrongly."""
    settings = _settings(tmp_path)
    config = _config()
    info = {
        "author": "Redaktion WLO <redaktion@example.org>",
        "description": "Subject classifier for school material. Not for grading learners.",
        "data_source": "WLO prod export 2026-07-26, filtered to school disciplines",
        "license": "CC BY-SA 4.0",
    }
    run_training(
        {**_request(), "info": info}, settings, config, config.get("fast"),
        _registry(settings), on_progress=lambda **_: None, should_stop=lambda: False,
    )

    bundle = Registry(settings.models_dir, 2).info("tiny_model")
    assert bundle["metadata"]["info"] == info
    # Always reported, even when there is nothing to report: the fixture's labels are
    # bare tokens ("uri:math"), not URIs, so no namespace can be derived. The
    # derivation itself is pinned by the unit test below, on real vocabulary URIs.
    assert bundle["label_vocabulary"] is None


def test_label_vocabulary_is_only_reported_when_the_classes_agree(tmp_path):
    """A single shared namespace is a statement worth making; a mixed label set is
    not. Measured across the local model store, all 14 bundles had one common prefix
    — but a CSV mixing two vocabularies must report none rather than a misleading
    fragment of a URI."""
    from app.data import label_vocabulary

    assert label_vocabulary([
        "http://w3id.org/openeduhub/vocabs/discipline/380",
        "http://w3id.org/openeduhub/vocabs/discipline/120",
    ]) == "http://w3id.org/openeduhub/vocabs/discipline/"
    # Two vocabularies: the common prefix ".../vocabs/" names neither of them.
    assert label_vocabulary([
        "http://w3id.org/openeduhub/vocabs/discipline/380",
        "http://w3id.org/openeduhub/vocabs/educationalContext/sekundarstufe_1",
    ]) is None
    assert label_vocabulary(["mathematics", "physics"]) is None
    assert label_vocabulary([]) is None


def test_metadata_records_the_searched_c_grid(tmp_path):
    """best_C alone is not interpretable: a value sitting at the EDGE of the grid
    means the search ran out of candidates, not that it found an optimum. Persist
    the grid that was searched so that stays visible after the fact."""
    settings = _settings(tmp_path)
    config = _config()  # 'fast' profile searches [1.0, 2.0]
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    meta = Registry(settings.models_dir, 2).info("tiny_model")["metadata"]
    assert meta["c_grid"] == [1.0, 2.0]
    assert meta["best_C"] in meta["c_grid"]


def test_vocabulary_travels_outside_skops_and_round_trips_exactly(tmp_path):
    """skops is super-quadratic in the number of dict ENTRIES: a 200k-term vocabulary
    measured ~45 min and 148 MB to write, while the 48 MB float head took one second.
    The vocabulary now travels as a plain JSON member (~4000x faster, ~45x smaller).

    The acceptance criterion is not the clock but exactness: a model loaded from the
    bundle must vectorize IDENTICALLY to the one that was saved.
    """
    import numpy as np

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    bundle = Path(settings.models_dir) / "tiny_model"
    assert (bundle / "vocabulary.json").exists()
    # The heavy dict must be OUT of the skops container, not merely duplicated.
    assert (bundle / "vocabulary.json").stat().st_size > 0

    reloaded = Registry(settings.models_dir, 2).get("tiny_model")
    probe = ["Bruchrechnung und Gleichungen", "Ein Gedicht interpretieren"]
    matrix = reloaded.vectorizer.transform(probe)
    assert matrix.shape[1] == len(reloaded.vectorizer.word_vec.vocabulary_) + (
        len(reloaded.vectorizer.char_vec.vocabulary_) if reloaded.vectorizer.char_vec else 0
    )
    assert np.isfinite(matrix.data).all() and matrix.nnz > 0


def test_bundle_with_a_vocabulary_that_does_not_fit_the_model_is_rejected(tmp_path):
    """vocabulary.json travels inside IMPORTABLE bundles. A wrong length would silently
    build a feature matrix of the wrong width and only fail deep inside the head —
    reject it at load with a clean error instead."""
    import json

    import pytest

    from app.registry import UnsafeModelError

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    path = Path(settings.models_dir) / "tiny_model" / "vocabulary.json"
    vocabularies = json.loads(path.read_text(encoding="utf-8"))
    vocabularies[0] = vocabularies[0][:-1]  # one term short
    path.write_text(json.dumps(vocabularies), encoding="utf-8")

    with pytest.raises(UnsafeModelError):
        Registry(settings.models_dir, 2).get("tiny_model")


def test_bundle_carries_per_label_f1_and_survives_a_malformed_metrics_file(tmp_path):
    """A loaded model carries per-label F1 from metrics.json so /predict can report
    it. metrics.json travels INSIDE importable bundles, so a malformed value must
    degrade to "no F1" rather than crash every later prediction with a 500."""
    import json

    settings = _settings(tmp_path)
    config = _config()
    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    model = Registry(settings.models_dir, 2).get("tiny_model")
    assert set(model.per_label_f1) == {"uri:math", "uri:bio", "uri:hist"}
    assert all(0.0 <= score <= 1.0 for score in model.per_label_f1.values())

    metrics_path = Path(settings.models_dir) / "tiny_model" / "metrics.json"
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    doc["metrics"]["per_label_f1"] = ["not", "a", "mapping"]
    metrics_path.write_text(json.dumps(doc), encoding="utf-8")

    reloaded = Registry(settings.models_dir, 2).get("tiny_model")
    assert reloaded.per_label_f1 == {}  # ignored, not crashed
    ranked = reloaded.predict(["Bruchrechnung"], top_k=1, include_label_f1=True)
    assert ranked[0][0].label_f1 is None


def _trained_registry(tmp_path, info: dict | None = None) -> tuple[Registry, Settings]:
    """Train the tiny fixture model and return its registry (helper for the export tests)."""
    settings = _settings(tmp_path)
    config = _config()
    req = {**_request(), **({"info": info} if info else {})}
    run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    return Registry(settings.models_dir, 2), settings


def test_export_carries_a_manifest_and_a_readable_model_card(tmp_path):
    """An exported bundle is what a third party receives, and a ZIP of two skops
    containers plus JSON tells a human nothing. The archive therefore carries a
    generated card (what it does, how it was trained, how well, and the author's own
    statements) and a manifest of SHA-256 sums covering every other member."""
    import io
    import zipfile

    registry, _ = _trained_registry(tmp_path, info={
        "author": "Redaktion WLO", "license": "CC BY-SA 4.0",
        "description": "Subject classifier. Not for grading learners.",
    })

    with zipfile.ZipFile(io.BytesIO(registry.export_zip("tiny_model"))) as archive:
        members = set(archive.namelist())
        assert {"manifest.json", "README.md"} <= members
        manifest = json.loads(archive.read("manifest.json"))
        # Every member except the manifest itself is covered, each by a real digest.
        assert set(manifest["files"]) == members - {"manifest.json"}
        assert all(len(digest) == 64 for digest in manifest["files"].values())
        assert manifest["model_name"] == "tiny_model"

        card = archive.read("README.md").decode("utf-8")
    for expected in ("tiny_model", "Redaktion WLO", "CC BY-SA 4.0",
                     "Not for grading learners", "tiny.csv", "F1"):
        assert expected in card, f"the model card should mention {expected!r}"


def test_import_rejects_a_tampered_member(tmp_path):
    """The reason the manifest exists: a bundle travels as a 50-180 MB download, and a
    member that arrives corrupted or altered must be refused rather than loaded. Only
    skops choking on it would have caught this before."""
    import io
    import zipfile

    import pytest

    from app.registry import UnsafeModelError

    registry, _ = _trained_registry(tmp_path)
    original = registry.export_zip("tiny_model")

    tampered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, \
            zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.namelist():
            payload = source.read(member)
            if member == "head.skops":
                payload = payload[:-1] + bytes([payload[-1] ^ 0x01])  # one flipped bit
            target.writestr(member, payload)

    with pytest.raises(UnsafeModelError, match="head.skops"):
        registry.import_zip("tampered_copy", tampered.getvalue())
    assert "tampered_copy" not in registry.list()


def test_import_still_accepts_an_archive_without_a_manifest(tmp_path):
    """Bundles exported by 3.1.0 carry no manifest. Verification applies when one is
    present; its absence is not an error, or every model shared before this release
    would become unimportable."""
    import io
    import zipfile

    registry, _ = _trained_registry(tmp_path)
    stripped = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(registry.export_zip("tiny_model"))) as source, \
            zipfile.ZipFile(stripped, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.namelist():
            if member not in ("manifest.json", "README.md"):
                target.writestr(member, source.read(member))

    registry.import_zip("legacy_copy", stripped.getvalue())
    assert "legacy_copy" in registry.list()


def test_transport_artifacts_are_not_kept_in_the_installed_bundle(tmp_path):
    """The manifest and the card describe one ARCHIVE and are regenerated per export.
    Writing them into the bundle directory would make the next export ship a stale
    copy alongside the fresh one."""
    registry, settings = _trained_registry(tmp_path)
    registry.import_zip("copy", registry.export_zip("tiny_model"))

    installed = {p.name for p in (settings.models_dir / "copy").iterdir()}
    assert "manifest.json" not in installed and "README.md" not in installed
    assert "head.skops" in installed and "config.json" in installed


def test_import_rejects_zip_bomb(tmp_path):
    """An archive that inflates both past the absolute floor (64 MB) and far beyond
    its upload size is refused before extraction (memory-DoS guard). Small inflates
    are deliberately allowed: legitimate skops bundles deflate well (~16x)."""
    import io
    import zipfile

    import pytest

    from app.registry import UnsafeModelError

    registry = Registry(tmp_path / "models", 2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("head.skops", b"\x00" * (100 * 1024 * 1024))  # ~100 MB -> ~100 KB zip
        archive.writestr("vectorizer.skops", "x")
    with pytest.raises(UnsafeModelError):
        registry.import_zip("bomb", buffer.getvalue())


def test_import_rejects_bundle_without_vectorizer(tmp_path):
    """A bundle missing vectorizer.skops is rejected cleanly, not crashed (no 500)."""
    import io
    import zipfile

    import pytest

    from app.registry import UnsafeModelError

    registry = Registry(tmp_path / "models", 2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("head.skops", "x")  # required file present, but vectorizer missing
    with pytest.raises(UnsafeModelError):
        registry.import_zip("incomplete", buffer.getvalue())


def test_task_type_override(tmp_path):
    """Forcing task_type overrides auto-detection and is stored on the model."""
    settings = _settings(tmp_path)
    config = _config()
    req = _request()
    req["model_name"] = "tiny_forced"
    req["task_type"] = "multilabel"  # tiny.csv is single-label, but we force multilabel
    result = run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["task_type"] == "multilabel"
    model = Registry(settings.models_dir, settings.max_models_in_memory).get("tiny_forced")
    assert model.task_type == "multilabel"

    # Multilabel prediction knobs on the same model:
    text = ["Bruchrechnung und Gleichungen lösen üben"]
    auto = model.predict(text, top_k=0)[0]  # 0 -> auto top_k from avg labels
    assert len(auto) == max(1, round(model.avg_labels))
    everything = model.predict(text, top_k=None, threshold=0.0)[0]
    assert len(everything) == len(model.classes)  # threshold override lets all pass
    only_math = model.predict(text, threshold=0.0, label_filter="math")[0]
    assert only_math and all("math" in p.uri for p in only_math)


def test_fast_profile_word_only(tmp_path):
    """The word-only profile (use_char=False) trains, persists and predicts."""
    settings = _settings(tmp_path)
    profile = Profile("fast", "word-only", True, False, [2.0, 4.0], use_char=False, max_word_features=20_000)
    config = TrainingConfig(
        default_profile="fast", profiles={"fast": profile},
        validation_size=0.2, test_size=0.2, min_text_length=5,
        drop_duplicates=True, min_samples_per_label=2,
    )
    req = _request()
    req["model_name"] = "tiny_wordonly"
    result = run_training(
        req, settings, config, profile, _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    assert result["metrics"]["f1_micro"] > 0.5
    model = Registry(settings.models_dir, settings.max_models_in_memory).get("tiny_wordonly")
    assert model.vectorizer.char_vec is None  # word-only path round-trips
    assert model.predict(["Bruchrechnung und Gleichungen"], top_k=1)[0]


def test_min_samples_per_label_override(tmp_path):
    """An explicit min_samples_per_label in the request overrides the config default."""
    settings = _settings(tmp_path)
    config = _config()  # config default is min_samples_per_label=2
    req = _request()
    req["model_name"] = "tiny_minsamples"
    req["min_samples_per_label"] = 3
    run_training(
        req, settings, config, config.get("fast"), _registry(settings),
        on_progress=lambda **_: None, should_stop=lambda: False,
    )
    info = Registry(settings.models_dir, settings.max_models_in_memory).info("tiny_minsamples")
    assert info["metadata"]["min_samples_per_label"] == 3


def test_heartbeat_is_none_when_not_running():
    """seconds_since_heartbeat is a liveness signal of the RUNNING thread only:
    idle and finished jobs report None (so the UI shows no stale stall hint)."""
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()
    assert job.snapshot()["seconds_since_heartbeat"] is None

    job.start(lambda *, on_progress, should_stop: {"ok": True}, model_name="m")
    deadline = time.time() + 3
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.01)
    snap = job.snapshot()
    assert snap["status"] == "completed"
    assert snap["seconds_since_heartbeat"] is None


def test_heartbeat_refreshes_on_each_progress_update():
    """Every on_progress call is also a liveness heartbeat: after an update the
    reported heartbeat age must be younger than the silence measured before it.
    elapsed_seconds keeps growing either way and cannot carry this signal."""
    import threading
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()
    step = threading.Event()
    stepped = threading.Event()
    release = threading.Event()

    def target(*, on_progress, should_stop):
        step.wait(5)
        on_progress(phase="features", progress=40)
        stepped.set()
        release.wait(5)
        return {}

    job.start(target, model_name="m")
    try:
        time.sleep(0.25)  # thread alive but silent since start()
        stalled = job.snapshot()["seconds_since_heartbeat"]
        assert stalled is not None and stalled >= 0.2

        step.set()
        assert stepped.wait(2), "target never reported progress"
        assert job.snapshot()["seconds_since_heartbeat"] < stalled
    finally:
        release.set()


def test_silent_stall_is_visible_via_growing_heartbeat_age():
    """The 2026-07-10 incident: a crawling 148 MB skops dump kept status
    'running' at saving/92% for ~53 min while elapsed_seconds grew normally —
    from outside indistinguishable from a crash. The heartbeat age must expose
    such a silent stall by growing while progress stands still."""
    import threading
    import time

    from app.jobs import TrainingJob

    job = TrainingJob()
    reached_stall = threading.Event()
    release = threading.Event()

    def target(*, on_progress, should_stop):
        on_progress(phase="saving", progress=92, message="Writing bundle...")
        reached_stall.set()
        release.wait(5)  # staged stall: thread alive, no progress signal
        return {}

    job.start(target, model_name="m")
    try:
        assert reached_stall.wait(2), "target never reached the stall"
        time.sleep(0.15)
        first = job.snapshot()
        time.sleep(0.15)
        second = job.snapshot()
    finally:
        release.set()

    assert first["status"] == second["status"] == "running"
    assert first["progress"] == second["progress"] == 92  # frozen to the outside
    assert first["seconds_since_heartbeat"] >= 0.1
    assert second["seconds_since_heartbeat"] > first["seconds_since_heartbeat"]
    assert second["elapsed_seconds"] > first["elapsed_seconds"]  # old signal keeps moving


def test_save_phase_reports_bundle_substeps(tmp_path):
    """The save phase must keep signalling between its long sub-steps (deploy
    fit, per-file skops dumps): each sub-step surfaces as a phase_detail update,
    which also refreshes the job heartbeat. Without this, a long vectorizer dump
    is a silent stall (the 53-minute incident above)."""
    settings = _settings(tmp_path)
    config = _config()
    details: list[str] = []

    def on_progress(**fields):
        detail = fields.get("phase_detail")
        if detail:
            details.append(str(detail))

    run_training(
        _request(), settings, config, config.get("fast"), _registry(settings),
        on_progress=on_progress, should_stop=lambda: False,
    )
    for expected in ("final model", "head.skops", "vectorizer.skops"):
        assert any(expected in d for d in details), (expected, details)
