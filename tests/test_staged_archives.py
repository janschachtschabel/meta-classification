"""Downloads of one bundle share one staged archive (audit 2026-09-30, S02).

An export -- an admin's, or anyone's through a public share link -- is packed into a file
beside the bundles and streamed from there, because a production bundle is too large to
build in memory. Each download used to pack its own full copy and keep it until its client
had read the last byte: twelve connections that never read held twelve copies, and about
twenty filled the chart's volume, after which every write failed. Two downloads of the
same bundle state are the same bytes, so what is pinned here is that they share one file,
that the last download to finish removes it, and that a changed bundle is packed anew.
"""

import json
import zipfile
from pathlib import Path

import pytest

from app import model_archive
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def registry(tmp_path) -> Registry:
    """A registry holding one small trained bundle, "m"."""
    settings = Settings(data_dir=FIXTURES, models_dir=tmp_path / "models", auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0])
    config = TrainingConfig(
        default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
        test_size=0.2, min_text_length=5, drop_duplicates=True, min_samples_per_label=2,
    )
    request = {
        "dataset_name": "tiny.csv", "model_name": "m",
        "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
        "label_column": "properties.ccm:taxonid", "csv_separator": ";",
        "label_separator": ",", "label_filter": None,
    }
    run_training(request, settings, config, profile, Registry(settings.models_dir, 2),
                 on_progress=lambda **_: None, should_stop=lambda: False)
    return Registry(settings.models_dir, 2)


def _staged(registry: Registry) -> list[Path]:
    return sorted(registry.dir.glob(".export-*"))


def test_downloads_of_one_bundle_share_one_staged_archive(registry):
    first = registry.stage_export("m")
    second = registry.stage_export("m")

    assert len(_staged(registry)) == 1, "each download packed its own copy"
    (path, release_first), (same_path, release_second) = first, second
    assert same_path == path
    release_first()
    assert path.exists(), "the other download is still reading it"
    release_second()
    assert not _staged(registry), "the last download to finish removes it"


def test_a_release_counts_once(registry):
    """A response that both fails and gets cleaned up must not take another download's
    file away from it."""
    (path, release_first), (_, release_second) = registry.stage_export("m"), registry.stage_export("m")

    release_first()
    release_first()

    assert path.exists()
    release_second()
    assert not path.exists()


def test_a_changed_bundle_is_packed_anew(registry):
    """An `update_info` changes what the archive holds, so a download that starts after it
    gets the new state -- while one still reading the old archive keeps it."""
    old_path, release_old = registry.stage_export("m")
    registry.update_info("m", {"author": "Redaktion"})

    new_path, release_new = registry.stage_export("m")

    assert new_path != old_path and old_path.exists()
    with zipfile.ZipFile(new_path) as archive:
        assert json.loads(archive.read("metrics.json"))["info"] == {"author": "Redaktion"}
    release_old()
    release_new()
    assert not _staged(registry)


def test_a_failed_packing_leaves_no_file_and_no_stale_entry(registry, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(model_archive, "pack_into", broken)
        with pytest.raises(OSError, match="disk full"):
            registry.stage_export("m")
    assert not _staged(registry)

    path, release = registry.stage_export("m")
    with zipfile.ZipFile(path) as archive:
        assert "head.skops" in archive.namelist(), "the next download packs a whole archive"
    release()


def test_an_unknown_bundle_stages_nothing(registry):
    with pytest.raises(FileNotFoundError):
        registry.stage_export("ghost")
    assert not _staged(registry)
