"""A published bundle survives a power cut (audit 2026-09-30, R13).

A rename is atomic but not durable on its own: without an fsync, a file's blocks can still be
in the page cache when the rename reaches the disk, and after a power cut the name points at
an empty file -- ext4 with delayed allocation does exactly that for new files. Nothing was
fsync'd, so a published bundle could come back with empty skops files. The guarantee is an
ORDER, so that is what is pinned: the data and the staging directory are synced before the
rename makes the bundle visible, and the directory holding it after.
"""

import os
from pathlib import Path

import pytest

from app import durability
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"
MEMBERS = {"config.json", "metrics.json", "head.skops", "vectorizer.skops", "vocabulary.json"}


@pytest.fixture
def events(monkeypatch) -> list[tuple[str, str]]:
    """What was synced and renamed, in order, by name."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(durability, "sync_file", lambda path: seen.append(("file", Path(path).name)))
    monkeypatch.setattr(durability, "sync_dir", lambda path: seen.append(("dir", Path(path).name)))
    real_replace = os.replace

    def replace(src, dst):
        seen.append(("rename", Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return seen


def _train(models_dir: Path) -> Registry:
    settings = Settings(data_dir=FIXTURES, models_dir=models_dir, auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0])
    config = TrainingConfig(default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
                            test_size=0.2, min_text_length=5, drop_duplicates=True,
                            min_samples_per_label=2)
    registry = Registry(models_dir, 2)
    run_training({"dataset_name": "tiny.csv", "model_name": "m",
                  "text_columns": ["properties.cclom:title"], "label_column": "properties.ccm:taxonid",
                  "csv_separator": ";", "label_separator": ",", "label_filter": None},
                 settings, config, profile, registry,
                 on_progress=lambda **_: None, should_stop=lambda: False)
    return registry


def _before_and_after(events: list[tuple[str, str]], renamed: str) -> tuple[list, list]:
    at = events.index(("rename", renamed))
    return events[:at], events[at + 1:]


def test_a_trained_bundle_is_on_disk_before_its_name_is(tmp_path, events):
    _train(tmp_path / "models")

    before, after = _before_and_after(events, "m")
    assert {name for kind, name in before if kind == "file"} >= MEMBERS
    assert ("dir", ".m.tmp") in before, "the staging directory's entries too"
    assert ("dir", "models") in after, "and the rename itself, in the directory holding it"


def test_an_imported_bundle_is_on_disk_before_its_name_is(tmp_path, events):
    registry = _train(tmp_path / "models")
    archive = tmp_path / "m.zip"
    with archive.open("wb") as sink:
        registry.export_to("m", sink)
    events.clear()

    registry.import_archive("copy", archive)

    before, after = _before_and_after(events, "copy")
    assert {name for kind, name in before if kind == "file"} >= MEMBERS
    assert ("dir", ".copy.tmp") in before
    assert ("dir", "models") in after


def test_an_edited_metrics_file_is_on_disk_before_it_replaces_the_old_one(tmp_path, events):
    registry = _train(tmp_path / "models")
    events.clear()

    registry.update_info("m", {"author": "Redaktion"})

    before, after = _before_and_after(events, "metrics.json")
    assert ("file", "metrics.json.tmp") in before
    assert ("dir", "m") in after


def test_syncing_a_tree_fsyncs_each_file_and_where_possible_the_directory(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    for name in ("a", "b", "c"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    durability.sync_tree(tmp_path)

    # Windows cannot open a directory to fsync it; NTFS journals the entries itself.
    assert len(synced) == (3 if os.name == "nt" else 4)
