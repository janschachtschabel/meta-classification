"""Staging a bundle apart from publishing it.

A training in a child process writes its bundle while the API keeps serving; the API
process publishes it under its own disk lock. What is pinned here: a staged bundle is
invisible until it is published, publishing never replaces a name that appeared in the
meantime, and a staged bundle nobody publishes can be removed without a trace.
"""

import shutil
from pathlib import Path

import pytest

from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"


def _trained_model(tmp_path):
    """A small fitted model and its metadata, from one run of the fixture."""
    settings = Settings(data_dir=FIXTURES, models_dir=tmp_path / "source", auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0, 2.0])
    config = TrainingConfig(
        default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
        test_size=0.2, min_text_length=5, drop_duplicates=True, min_samples_per_label=2,
    )
    request = {
        "dataset_name": "tiny.csv", "model_name": "tiny_model",
        "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
        "label_column": "properties.ccm:taxonid", "csv_separator": ";",
        "label_separator": ",", "label_filter": None,
    }
    registry = Registry(settings.models_dir, 2)
    run_training(request, settings, config, profile, registry,
                 on_progress=lambda **_: None, should_stop=lambda: False)
    return registry.load_fresh("tiny_model")


def _hidden(registry: Registry) -> list[Path]:
    return [p for p in registry.dir.iterdir() if p.name.startswith(".")]


def test_a_staged_bundle_stays_invisible_until_it_is_published(tmp_path):
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)

    staged = registry.stage("staged", model, metadata)
    assert not registry.exists("staged") and "staged" not in registry.list()

    registry.publish("staged", staged)
    assert registry.exists("staged") and not _hidden(registry)
    assert registry.get("staged").classes == model.classes


def test_publishing_refuses_a_name_that_appeared_meanwhile_and_cleans_up(tmp_path):
    """An import can take the name while the child stages. The staged bundle must not
    replace it, and must not be left behind."""
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    staged = registry.stage("taken", model, metadata)
    shutil.copytree(staged, registry.dir / "taken")  # a bundle took the name meanwhile

    with pytest.raises(FileExistsError):
        registry.publish("taken", staged)
    assert not _hidden(registry)


def test_discarding_removes_a_staged_bundle(tmp_path):
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    staged = registry.stage("dropped", model, metadata)
    registry.discard_staged(staged)
    assert not _hidden(registry)
    assert not registry.exists("dropped")



# --- R11: a delete cut short does not block the name --------------------------------------------


def test_a_delete_cut_short_leaves_the_name_free_and_nothing_visible(tmp_path, monkeypatch):
    """R11 (audit 2026-09-30): delete was one rmtree. Cut short -- a file another process holds,
    an I/O error -- it left part of the bundle under the model's name: no longer a model, but
    in the way, so an import under that name failed with a 500."""
    import contextlib

    from app import registry as registry_mod

    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.save("m", model, metadata)
    archive = tmp_path / "m.zip"
    with archive.open("wb") as sink:
        registry.export_to("m", sink)

    def cut_short(path, *args, **kwargs):
        next(p for p in Path(path).iterdir() if p.is_file()).unlink()
        raise OSError("a file in it is in use")

    with monkeypatch.context() as patched:
        patched.setattr(registry_mod.shutil, "rmtree", cut_short)
        with contextlib.suppress(OSError):
            registry.delete("m")

    assert not registry.exists("m") and "m" not in registry.list()
    assert [p.name for p in registry.dir.iterdir() if not p.name.startswith(".")] == [], (
        "what is left is hidden, not lying under the name")
    registry.import_archive("m", archive)
    assert registry.get("m").classes == model.classes, "the name took a new bundle"
    assert registry.sweep_stale_tmp() == 1, "and the next start removes the leftover"



# --- R01: one staging directory per operation ----------------------------------------------------


def test_a_failing_import_leaves_a_trainings_staged_bundle_alone(tmp_path):
    """R01 (audit 2026-09-30): an import and a training of one name both staged in
    `.{name}.tmp`, and each removed what it found there as a crash leftover. An import that
    failed took the training's finished bundle with it -- the training's publish then had
    nothing to publish -- and two that overlapped published a mix (the audit's: 200 on import,
    "X has 182 features" on every prediction)."""
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    staged = registry.stage("m", model, metadata)  # a training of "m", about to publish
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip archive")

    with pytest.raises(Exception):  # noqa: B017 - which refusal is model_archive's business
        registry.import_archive("m", broken)  # an import of "m" that fails meanwhile
    registry.publish("m", staged)

    assert registry.get("m").classes == model.classes


def test_two_stagings_of_one_name_get_two_directories(tmp_path):
    """Neither can remove or overwrite what the other writes; whichever publishes second is
    refused, as a name taken meanwhile always was."""
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)

    first = registry.stage("m", model, metadata)
    second = registry.stage("m", model, metadata)

    assert first != second and first.exists() and second.exists()
    registry.publish("m", first)
    with pytest.raises(FileExistsError):
        registry.publish("m", second)
    assert not second.exists(), "the refused one's staging is removed"



def test_an_import_reserves_its_name_until_it_is_done(tmp_path):
    """R01's other half: an import's upload takes minutes, and nothing else could see it
    coming -- a training of the same name was accepted meanwhile, and whichever published
    second failed after all its work. The name is reserved for the import's whole length."""
    registry = Registry(tmp_path / "models", 2)

    with registry.importing("m"):
        assert registry.is_importing("m")
        with pytest.raises(FileExistsError), registry.importing("m"):
            pass
    assert not registry.is_importing("m"), "released however the import ended"

    with pytest.raises(RuntimeError), registry.importing("m"):
        raise RuntimeError("the upload broke off")
    assert not registry.is_importing("m")



def test_a_bundle_keeps_its_newest_evaluations_not_every_one_ever(tmp_path):
    """R15 (audit 2026-09-30): every evaluation appended to `metrics.json` for good -- the
    document every model detail reads and every export packs grew without end. The newest
    MAX_EVALUATIONS stay: enough to compare a model across datasets, which is their point."""
    import json

    from app.registry import MAX_EVALUATIONS

    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.save("m", model, metadata)

    for run in range(MAX_EVALUATIONS + 10):
        registry.append_evaluation("m", {"dataset": f"run-{run}.csv"})

    stored = json.loads((registry.dir / "m" / "metrics.json").read_text(encoding="utf-8"))
    kept = [entry["dataset"] for entry in stored["evaluations"]]
    assert kept == [f"run-{run}.csv" for run in range(10, MAX_EVALUATIONS + 10)]



def test_a_bundle_rewritten_on_disk_is_served_as_rewritten(tmp_path):
    """R15 (audit 2026-09-30): the label repairs (scripts/patch_bundle_labels.py,
    prune_bundle_labels.py) rewrite a bundle while the server runs, and the server kept
    serving the model it had cached -- until a restart or an eviction. A bundle removed by
    hand kept answering the same way."""
    import json

    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.save("m", model, metadata)
    cached = registry.get("m")
    config_path = registry.dir / "m" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["uri_to_label"] = {uri: f"Umbenannt {uri}" for uri in config["classes"]}
    config_path.write_text(json.dumps(config), encoding="utf-8")  # as patch_bundle_labels does

    reloaded = registry.get("m")

    assert reloaded is not cached
    assert reloaded.uri_to_label[config["classes"][0]].startswith("Umbenannt")
    assert registry.get("m") is reloaded, "and cached again until the next change"
    shutil.rmtree(registry.dir / "m")
    with pytest.raises(FileNotFoundError):
        registry.get("m")



def test_a_replace_that_fails_midway_keeps_the_old_bundle_serving(tmp_path, monkeypatch):
    """W04 (audit 2026-09-30): replacing a bundle -- what the label repair does to a serving
    model -- removed the old one first and renamed the new one in after. An error between the
    two left no model at all."""
    import app.registry as registry_module

    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.save("m", model, {**metadata, "marker": "old"})
    staged = registry.stage("m", model, {**metadata, "marker": "new"}, overwrite=True)
    real_replace = registry_module.os.replace

    def replace(source, destination):
        if Path(source) == staged:
            raise OSError("the disk went away")
        return real_replace(source, destination)

    monkeypatch.setattr(registry_module.os, "replace", replace)
    with pytest.raises(OSError):
        registry.publish("m", staged, overwrite=True)
    monkeypatch.undo()

    assert registry.exists("m"), "the model is gone"
    assert registry.load_fresh("m")[1]["marker"] == "old"


def test_a_replace_killed_between_its_renames_is_undone_at_the_next_start(tmp_path):
    """The window no handler covers: the process killed after the old bundle was renamed aside
    and before the new one was renamed in. The next start puts the old one back; the sweep
    used to delete both hidden directories -- and the model with them."""
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.save("m", model, {**metadata, "marker": "old"})
    staged = registry.stage("m", model, {**metadata, "marker": "new"}, overwrite=True)
    (registry.dir / "m").rename(registry.dir / ".m.replaced-0123abcd.tmp")

    restarted = Registry(tmp_path / "models", 2)
    restarted.sweep_stale_tmp()

    assert restarted.exists("m"), "the interrupted replace lost the model"
    assert restarted.load_fresh("m")[1]["marker"] == "old"
    assert not staged.exists() and not _hidden(restarted)
