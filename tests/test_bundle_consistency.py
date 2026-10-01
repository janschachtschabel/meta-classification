"""A bundle's members must belong together (audit 2026-09-30, R08).

A bundle is importable, so its five files can come from different runs. A head fitted on
another vectorizer's features installed with 200, and every prediction then failed with a 500
("X has 182 features, but LogisticRegression is expecting 7"). The loader -- which an import
runs before publishing -- now checks that the head takes the features the vectorizer makes
and scores the classes the config names.
"""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from app.errors import UnsafeModelError
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"
MEMBERS = ("config.json", "metrics.json", "head.skops", "vectorizer.skops", "vocabulary.json")


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> Path:
    """Two bundles on the same labels whose vectorizers make different numbers of features."""
    models = tmp_path_factory.mktemp("bundles")
    settings = Settings(data_dir=FIXTURES, models_dir=models, auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0])
    config = TrainingConfig(default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
                            test_size=0.2, min_text_length=5, drop_duplicates=True,
                            min_samples_per_label=2)
    for name, columns in (("narrow", ["properties.cclom:title"]),
                          ("wide", ["properties.cclom:title", "properties.cclom:general_keyword"])):
        run_training({"dataset_name": "tiny.csv", "model_name": name, "text_columns": columns,
                      "label_column": "properties.ccm:taxonid", "csv_separator": ";",
                      "label_separator": ",", "label_filter": None},
                     settings, config, profile, Registry(models, 2),
                     on_progress=lambda **_: None, should_stop=lambda: False)
    return models


def test_the_two_bundles_really_differ(bundles):
    """What the tests below rest on."""
    registry = Registry(bundles, 2)
    narrow, wide = registry.load_fresh("narrow")[0], registry.load_fresh("wide")[0]
    assert narrow.head.n_features_in_ != wide.head.n_features_in_


def test_an_archive_with_another_bundles_head_is_refused_on_import(bundles, tmp_path):
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in MEMBERS:
            source = "wide" if member == "head.skops" else "narrow"
            zipped.write(bundles / source / member, member)
    registry = Registry(tmp_path / "models", 2)

    with pytest.raises(UnsafeModelError, match="features"):
        registry.import_archive("mixed", archive)

    assert registry.list() == [], "nothing was installed"


def test_a_bundle_on_disk_whose_parts_disagree_is_refused_on_load(bundles, tmp_path):
    """A bundle edited on the volume, or installed before this check, answers 422 -- not a
    500 on every prediction."""
    models = tmp_path / "models"
    shutil.copytree(bundles / "narrow", models / "mixed")
    shutil.copy(bundles / "wide" / "head.skops", models / "mixed" / "head.skops")

    with pytest.raises(UnsafeModelError, match="features"):
        Registry(models, 2).load_fresh("mixed")


def test_a_config_naming_other_classes_than_the_head_scores_is_refused(bundles, tmp_path):
    models = tmp_path / "models"
    shutil.copytree(bundles / "narrow", models / "mixed")
    config_path = models / "mixed" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["classes"] = config["classes"][:-1]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(UnsafeModelError, match="classes"):
        Registry(models, 2).load_fresh("mixed")


def test_a_consistent_bundle_still_loads(bundles):
    model, _ = Registry(bundles, 2).load_fresh("wide")
    assert model.predict_proba(["Bruchrechnung"]).shape == (1, len(model.classes))
