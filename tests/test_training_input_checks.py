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



# --- T03: a text column the CSV does not have -------------------------------------------------


def test_a_text_column_the_csv_lacks_is_refused_not_skipped(tmp_path):
    """T03: the loader skipped it, the run completed, and the bundle recorded the column
    anyway. `/predict/csv` then refused a CSV in the training data's own format, and a
    later export that has the column would feed the model text it never saw."""
    from app.dataset_load import load_dataset

    rows = [f"{text} Teil {i};{uri}" for uri, text in SUBJECTS.items() for i in range(5)]
    _dataset(tmp_path, rows)

    with pytest.raises(ValueError, match="beschreibung"):
        load_dataset(tmp_path / "set.csv", ["title", "beschreibung"], "labels", separator=";")



# --- T04: the task type ----------------------------------------------------------------------


def test_the_task_type_is_that_of_the_labels_trained_not_of_the_raw_cells(tmp_path):
    """T04: `uri:math,uri:math` on one row, or a second label too rare to train, made a
    three-class dataset "multilabel" -- decided by thresholds instead of argmax, so a
    nonsense text got all three labels."""
    rows = [f"{text} Teil {i};{uri}" for uri, text in SUBJECTS.items() for i in range(20)]
    rows[0] = rows[0].replace(";uri:math", ";uri:math,uri:math")
    rows[1] = rows[1].replace(";uri:math", ";uri:math,uri:rare")  # 1 row < min_samples 2
    settings = _dataset(tmp_path, rows)

    result = _train(settings)

    assert result["task_type"] == "multiclass"


def test_a_label_named_twice_in_one_cell_counts_once(tmp_path):
    from app.dataset_load import load_dataset

    _dataset(tmp_path, [f"{text} Teil;uri:a,uri:a,uri:b" for text in SUBJECTS.values()])

    loaded = load_dataset(tmp_path / "set.csv", ["title"], "labels", separator=";")

    assert loaded.label_lists[0] == ["uri:a", "uri:b"]


def test_the_task_type_follows_the_target_matrix():
    assert data.detect_task_type(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) == "multiclass"
    assert data.detect_task_type(np.array([[1, 0], [0, 1]])) == "binary"
    assert data.detect_task_type(np.array([[1, 1], [1, 0]])) == "multilabel"



# --- T07, T10: duplicates ----------------------------------------------------------------------


def _loaded(tmp_path, rows, **kwargs):
    from app.dataset_load import load_dataset

    _dataset(tmp_path, rows)
    return load_dataset(tmp_path / "set.csv", ["title"], "labels", separator=";", **kwargs)


def test_a_duplicate_is_what_the_vectorizer_cannot_tell_apart(tmp_path):
    """T07: the dedupe compared exactly, while the vectorizer lower-cases and strips accents
    -- so "BRUCHRECHNUNG" stayed beside "Bruchrechnung", one could land in train and its twin
    in test, and the metrics looked better than the model is."""
    rows = ["Bruchrechnung für Einsteiger;uri:math", "BRUCHRECHNUNG FÜR EINSTEIGER;uri:math",
            "Bruchrechnung fur Einsteiger;uri:math", "Photosynthese im Blatt;uri:bio"]

    assert _loaded(tmp_path, rows).texts == ["Bruchrechnung für Einsteiger", "Photosynthese im Blatt"]
    assert len(_loaded(tmp_path, rows, drop_duplicates=False).texts) == 4


def test_a_duplicate_with_other_labels_is_counted_not_lost_in_silence(tmp_path):
    """T10: the first row wins -- merging would make a single-label dataset multilabel for a
    few noisy copies -- but the copies whose labels differ are counted and reported."""
    rows = ["Bruchrechnung für Einsteiger;uri:math", "Bruchrechnung für Einsteiger;uri:bio",
            "Photosynthese im Blatt;uri:bio", "Photosynthese im Blatt;uri:bio"]

    loaded = _loaded(tmp_path, rows)

    assert loaded.texts == ["Bruchrechnung für Einsteiger", "Photosynthese im Blatt"]
    assert loaded.conflicting_duplicates == 1


def test_the_bundle_reports_duplicates_whose_labels_disagreed(tmp_path):
    rows = [f"{text} Teil {i};{uri}" for uri, text in SUBJECTS.items() for i in range(20)]
    rows.append(rows[0].replace(";uri:math", ";uri:bio"))
    settings = _dataset(tmp_path, rows)

    _train(settings)

    _model, metadata = Registry(settings.models_dir, 2).load_fresh("m")
    assert metadata["conflicting_duplicates"] == 1



# --- T08: labels pandas would call missing -------------------------------------------------------


def test_a_label_called_na_is_a_label(tmp_path):
    """T08: pandas reads "NA", "None", "null" and "nan" as missing values, and the rows
    carrying such a label vanished. The mark columns were already read as written."""
    rows = ["Erster Text über Mathe;NA", "Zweiter Text über Mathe;None",
            "Dritter Text über Mathe;null", "Vierter Text über Mathe;nan",
            "Fünfter Text ohne Label;"]

    loaded = _loaded(tmp_path, rows)

    assert loaded.label_lists == [["NA"], ["None"], ["null"], ["nan"]]


def test_validation_does_not_count_a_label_called_na_as_missing(tmp_path):
    from app.dataset_stats import validate_dataset

    _dataset(tmp_path, ["Erster Text über Mathe;NA", "Zweiter Text über Mathe;uri:math"])

    report = validate_dataset(tmp_path / "set.csv", ["title"], "labels")

    assert not any("without labels" in warning for warning in report["warnings"]), report



# --- T09: a comparison is not a tag, for new models -------------------------------------------

_MATH = "Für alle x < y gilt: Wenn a > b, dann ist die Differenz positiv."


def test_a_comparison_survives_the_cleaning_a_new_model_is_trained_with():
    """T09: any `<...>` was a tag, so the vectorizer path made "Für alle x b, dann ..." of
    a sentence in maths material. A tag starts with `<` and a letter, `/`, `!` or `?`."""
    cleaned = data.clean_text(_MATH)

    assert "y gilt" in cleaned and "Wenn a" in cleaned, cleaned
    assert data.clean_text("5 &lt; 7 und 9 &gt; 2") == "5 < 7 und 9 2"


def test_a_model_trained_before_keeps_the_cleaning_it_was_trained_with():
    """Changing what the cleaning produces shifts the features a fitted vectorizer expects,
    so an existing bundle is served with the cleaning it was trained with."""
    assert data.clean_text(_MATH, version=1) == "Für alle x b, dann ist die Differenz positiv."


def test_the_cleaning_is_idempotent_in_both_versions():
    """Evaluation cleans twice (the loader, then the model); a second pass must not change
    what the first produced."""
    for version in (1, 2):
        for text in (_MATH, "&lt;b&gt;fett&lt;/b&gt; <p>Absatz</p> x < y", "a&lt;b und c&gt;d"):
            once = data.clean_text(text, version=version)
            assert data.clean_text(once, version=version) == once, (version, text)


def test_a_bundle_records_its_cleaning_and_an_old_one_is_served_with_version_1(tmp_path):
    import json

    from app.errors import UnsafeModelError

    rows = [f"{text} Teil {i};{uri}" for uri, text in SUBJECTS.items() for i in range(20)]
    settings = _dataset(tmp_path, rows)
    _train(settings)
    bundle = settings.models_dir / "m" / "config.json"
    config = json.loads(bundle.read_text(encoding="utf-8"))
    assert config["text_cleaning"] == data.CLEANING_VERSION

    del config["text_cleaning"]  # as every bundle written before the version existed
    bundle.write_text(json.dumps(config), encoding="utf-8")
    model, _ = Registry(settings.models_dir, 2).load_fresh("m")
    assert model.text_cleaning == 1

    config["text_cleaning"] = 99  # from a server newer than this one
    bundle.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(UnsafeModelError, match="cleaning"):
        Registry(settings.models_dir, 2).load_fresh("m")


def test_evaluation_reads_the_dataset_with_the_models_cleaning(tmp_path):
    from app.dataset_load import load_dataset

    _dataset(tmp_path, [f"{_MATH};uri:math"])

    old = load_dataset(tmp_path / "set.csv", ["title"], "labels", separator=";", text_cleaning=1)

    assert old.texts == ["Für alle x b, dann ist die Differenz positiv."]



# --- R15: a duration is not a difference of wall-clock readings ---------------------------------


def test_a_training_time_survives_the_wall_clock_being_set_back(tmp_path, monkeypatch):
    """R15 (audit 2026-09-30): the run took `time.time()` at its start and end, so a clock set
    back during it -- NTP after a suspend, a VM migration -- recorded a negative training time."""
    import itertools
    import time

    from app import training as training_mod

    now, steps = time.time(), itertools.count()
    # Every reading 10 s earlier than the one before: whichever two the run takes, the later
    # is the smaller. (Realistic values -- the ZIP writer stamps members with them.)
    monkeypatch.setattr(training_mod.time, "time", lambda: now - 10 * next(steps))
    rows = [f"{text} Teil {i};{uri}" for uri, text in SUBJECTS.items() for i in range(20)]
    settings = _dataset(tmp_path, rows)

    result = _train(settings)

    assert result["training_time_seconds"] >= 0
