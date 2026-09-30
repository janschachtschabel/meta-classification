"""What a training does with data it cannot learn from as asked (audit 2026-09-30, T-findings).

Each case here once ended in a run reported as `completed` whose model was broken or
silently different from the request: a label on every row gave a bundle the server refuses to
load (T01). The rule these tests hold: a run either trains on what it reports or refuses with
a message the operator can act on.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.multiclass import OneVsRestClassifier

from app import data
from app.classifier import ClassifierModel, make_head
from app.errors import UnsafeModelError
from app.prepare import _drop_unlearnable
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training
from app.vectorizers import TfidfBackend

SUBJECTS = {
    "uri:math": "Bruchrechnung mit Nennern und Zählern üben, Gleichungen umstellen",
    "uri:bio": "Photosynthese im Blatt, Zellatmung und Chlorophyll im Versuch",
    "uri:hist": "Französische Revolution, Quellenarbeit zur Erklärung der Menschenrechte",
}


def _config() -> TrainingConfig:
    profile = Profile("fast", "TF-IDF", True, True, [1.0, 2.0])
    return TrainingConfig(
        default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
        test_size=0.2, min_text_length=5, drop_duplicates=True, min_samples_per_label=2,
    )


def _dataset(tmp_path: Path, rows: list[str], name: str = "set.csv") -> Settings:
    (tmp_path / name).write_text("title;labels\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)


def _train(settings: Settings, name: str = "set.csv", **overrides) -> dict:
    config = _config()
    request = {
        "dataset_name": name, "model_name": "m", "text_columns": ["title"],
        "label_column": "labels", "csv_separator": ";", "label_separator": ",",
        "label_filter": None, **overrides,
    }
    return run_training(request, settings, config, config.get("fast"),
                        Registry(settings.models_dir, 2),
                        on_progress=lambda **_: None, should_stop=lambda: False)


# --- T01: a label on every row ----------------------------------------------------------------


def test_a_label_on_every_row_is_not_a_target():
    """Nothing to learn from: no row lacks it. sklearn stores a `_ConstantPredictor` for such
    a column, which is not on skops' trusted list."""
    y, classes, keep = data.prepare_targets([["p", "a"]] * 6 + [["p", "b"]] * 6, min_samples=2)

    assert classes == ["a", "b"]
    assert y.shape == (12, 2)
    assert keep.all()


def test_a_label_the_rare_drop_leaves_on_every_row_is_not_a_target_either():
    """`q` is missing from eight rows -- but only rows whose labels are all too rare to keep.
    Once those rows go, `q` is on every row that is left."""
    lists = [["q", "a"]] * 10 + [["q", "b"]] * 10 + [["r1"]] * 4 + [["r2"]] * 4

    y, classes, keep = data.prepare_targets(lists, min_samples=5)

    assert classes == ["a", "b"]
    assert y.shape == (20, 2)
    assert keep.tolist() == [True] * 20 + [False] * 8


def test_the_holdout_split_drops_a_label_without_a_negative_in_train():
    """The train split is what a head is fitted on: it needs both classes there."""
    texts = np.array([f"text {i}" for i in range(6)], dtype=object)
    y = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
    splits = (np.array([0, 1, 2]), np.array([3, 4]), np.array([5]))

    texts2, y2, classes, (train, val, test), _marks = _drop_unlearnable(
        texts, y, ["a", "b", "c"], splits)

    assert classes == ["b", "c"], "`a` is on every train row"
    assert list(texts2) == list(texts), "every row keeps a label"
    assert (train.tolist(), val.tolist(), test.tolist()) == ([0, 1, 2], [3, 4], [5])


def test_the_holdout_split_drops_what_an_orphaned_train_row_leaves_on_every_row():
    """Row 1 is `b`'s only train negative, and its only label is `a` -- which goes first."""
    texts = np.array([f"text {i}" for i in range(6)], dtype=object)
    y = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1],
                  [0, 1, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
    splits = (np.array([0, 1, 2, 3]), np.array([4]), np.array([5]))

    texts2, y2, classes, (train, val, test), _marks = _drop_unlearnable(
        texts, y, ["a", "b", "c", "d"], splits)

    assert classes == ["c", "d"]
    assert list(texts2) == ["text 0", "text 2", "text 3", "text 4", "text 5"]
    assert (train.tolist(), val.tolist(), test.tolist()) == ([0, 1, 2], [3], [4])
    assert (y2[train].sum(axis=0) < len(train)).all(), "a label on every train row survived"


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_a_label_on_every_row_trains_a_model_that_loads(tmp_path, cv_folds):
    """T01: the run said `completed`, and every `/predict` then answered 422."""
    rows = [f"{text} Teil {i};uri:all,{uri}" for uri, text in SUBJECTS.items() for i in range(20)]
    settings = _dataset(tmp_path, rows)

    _train(settings, cv_folds=cv_folds)

    model, metadata = Registry(settings.models_dir, 2).load_fresh("m")  # through the skops guard
    assert sorted(model.classes) == sorted(SUBJECTS)
    predicted = model.predict([SUBJECTS["uri:bio"]], top_k=1)[0][0].uri
    assert predicted == "uri:bio"
    assert metadata["ubiquitous_labels"] == ["uri:all"]


def test_the_model_card_names_the_labels_left_out():
    from app import model_card

    card = model_card.render("m", {"classes": ["uri:a"]}, {"ubiquitous_labels": ["uri:all"]}, None)

    assert "`uri:all`" in card


def test_every_label_on_every_row_is_refused_with_the_reason(tmp_path):
    rows = [f"{text} Teil {i};uri:a,uri:b" for text in SUBJECTS.values() for i in range(10)]
    settings = _dataset(tmp_path, rows)

    with pytest.raises(ValueError, match="every row"):
        _train(settings)


def test_a_bundle_the_server_would_refuse_is_never_published(tmp_path):
    """The safety net behind the rule above: whatever else makes sklearn store a type the
    skops guard refuses, the run fails instead of publishing a model nobody can load."""
    vectorizer = TfidfBackend(use_char=False)
    x = vectorizer.fit_transform(["eins zwei", "drei vier", "fünf sechs"] * 4)
    y = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0]] * 4, dtype=np.int8)
    head = make_head(1.0, n_jobs=1, solver="newton-cg")
    with pytest.warns(UserWarning, match="present in all training examples"):
        head.fit(x, y)
    assert isinstance(head, OneVsRestClassifier)
    model = ClassifierModel(vectorizer=vectorizer, head=head, classes=["a", "b", "c"],
                            task_type="multilabel", avg_labels=2.0, uri_to_label={},
                            global_threshold=0.5, per_label_thresholds={})
    registry = Registry(tmp_path / "models", 2)

    with pytest.raises(UnsafeModelError, match="_ConstantPredictor"):
        registry.save("broken", model, {})

    assert not registry.exists("broken")
    assert not any((tmp_path / "models").iterdir()), "the staging directory stays behind"
