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
