"""End-to-end training on the tiny fixture (TF-IDF backend, fully offline).

Exercises the full pipeline (load -> prepare -> split -> bake-off -> threshold
-> evaluate -> deploy -> save) plus the registry save/get and export/import
round-trip with the secure skops format.
"""

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
