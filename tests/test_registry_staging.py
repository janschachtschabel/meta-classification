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

    registry.stage("staged", model, metadata)
    assert not registry.exists("staged") and "staged" not in registry.list()

    registry.publish("staged")
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
        registry.publish("taken")
    assert not _hidden(registry)


def test_discarding_removes_a_staged_bundle(tmp_path):
    model, metadata = _trained_model(tmp_path)
    registry = Registry(tmp_path / "models", 2)
    registry.stage("dropped", model, metadata)
    registry.discard_staged("dropped")
    assert not _hidden(registry)
    assert not registry.exists("dropped")
