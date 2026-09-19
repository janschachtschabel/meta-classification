"""Preparing a run on a dataset whose rows an LLM partly wrote: they train, never validate."""

import numpy as np
import pytest

from app.prepare import prepare_data
from app.profiles import Profile, TrainingConfig
from app.provenance import ENRICHED, EXAMPLE, GENERATED
from app.settings import Settings

HEADER = ["title", "keywords", "subject", "generated_for", "example_for", "enriched_fields"]


def _rows() -> list[list[str]]:
    rows = [[f"Bruchrechnung Aufgabe {i}", f"brüche{i}", "A", "", "", ""] for i in range(12)]
    rows += [[f"Photosynthese Versuch {i}", f"licht{i}", "B", "", "", "keywords" if i < 3 else ""]
             for i in range(12)]
    # Label C: two real rows, both shown to the generator, and ten it wrote.
    rows += [[f"Kaiser Augustus Quelle {i}", f"rom{i}", "C", "", "C", ""] for i in range(2)]
    rows += [[f"Erzeugt Antike Text {i}", f"antike{i}", "C", "C", "", ""] for i in range(10)]
    return rows


def _write(tmp_path, rows, header=HEADER, name="marked.csv"):
    lines = [";".join(header)] + [";".join(row[:len(header)]) for row in rows]
    (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def _prepare(tmp_path, name, *, cv_folds, stratified=False, **req):
    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    config = TrainingConfig(
        default_profile="fast", profiles={"fast": Profile("fast", "TF-IDF", True, True, [1.0])},
        validation_size=0.2, test_size=0.2, min_text_length=5, drop_duplicates=True,
        min_samples_per_label=2,
    )
    request = {"dataset_name": name, "model_name": "m", "text_columns": ["title", "keywords"],
               "label_column": "subject", "csv_separator": ";", "label_separator": ",",
               "label_filter": None, **req}
    prep = prepare_data(request, settings, config, cv_folds=cv_folds, stratified=stratified,
                        on_progress=lambda **_: None, should_stop=lambda: False)
    assert prep is not None
    return prep


def _unscored(prep) -> list[str]:
    scored = prep.provenance.scored
    return [] if scored is None else [c for c, ok in zip(prep.classes, scored, strict=True) if not ok]


def test_a_holdout_split_validates_and_tests_on_real_rows_only(tmp_path):
    prep = _prepare(tmp_path, _write(tmp_path, _rows()), cv_folds=0)

    train_only = prep.provenance.train_only()
    assert not train_only[prep.val_idx].any() and not train_only[prep.test_idx].any()
    assert train_only[prep.train_idx].sum() == train_only.sum() == 15
    assert prep.provenance.fallback is None
    assert _unscored(prep) == ["C"], "C is trained, but no real row can validate it"


def test_the_marks_stay_on_their_rows_through_every_drop(tmp_path):
    """A label below the minimum goes first -- prepare_targets drops its row, and a mark
    array that did not follow would be one row off from here on."""
    rows = [["Einzelstück ohne Nachbarn", "solo", "Z", "", "", ""], *_rows()]

    prep = _prepare(tmp_path, _write(tmp_path, rows), cv_folds=0)

    marks = prep.provenance.marks
    assert len(marks) == len(prep.texts)
    for text, mark in zip(prep.texts, marks, strict=True):
        assert bool(mark & GENERATED) == text.startswith("Erzeugt"), text
        assert bool(mark & EXAMPLE) == text.startswith("Kaiser"), text


def test_k_fold_validates_on_the_real_rows_only(tmp_path):
    prep = _prepare(tmp_path, _write(tmp_path, _rows()), cv_folds=3)

    assert prep.provenance.validate.tolist() == (prep.provenance.marks == 0).tolist()
    assert int(prep.provenance.validate.sum()) == 36 - 15
    assert _unscored(prep) == ["C"]


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_a_dataset_without_marks_is_prepared_exactly_as_before(tmp_path, cv_folds):
    """No mark column, or every mark blank: nothing is train-only, and the split is the
    one this dataset always got."""
    plain = [row[:3] for row in _rows()]
    blank = [[*row, "", "", ""] for row in plain]
    without = _prepare(tmp_path, _write(tmp_path, plain, HEADER[:3], "plain.csv"), cv_folds=cv_folds)
    with_blank = _prepare(tmp_path, _write(tmp_path, blank, HEADER, "blank.csv"), cv_folds=cv_folds)

    for prep in (without, with_blank):
        assert prep.provenance is None
        assert prep.classes == without.classes
        for part in ("train_idx", "val_idx", "test_idx"):
            assert np.array_equal(getattr(prep, part), getattr(without, part))


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_too_few_real_rows_fall_back_to_every_row_and_say_so(tmp_path, cv_folds):
    """A pure Runs export has no real row at all. Refusing would end the workflow the
    Runs push exists for, so it trains — and the bundle says its numbers are not real."""
    generated = [[f"Erzeugt {lab} Text {i}", f"kw{i}", lab, lab, "", ""]
                 for lab in ("A", "B") for i in range(12)]

    prep = _prepare(tmp_path, _write(tmp_path, generated), cv_folds=cv_folds)

    assert prep.provenance.fallback is not None
    assert prep.provenance.validate is None and prep.provenance.scored is None
    assert len(prep.val_idx) and len(prep.test_idx)


def test_exclude_drops_the_generated_rows_and_frees_the_examples(tmp_path):
    """k-fold, where every real row validates: the former examples must be among them."""
    prep = _prepare(tmp_path, _write(tmp_path, _rows()), cv_folds=3, synthetic_rows="exclude")

    assert not any(text.startswith("Erzeugt") for text in prep.texts)
    assert prep.provenance.excluded_generated == 10
    assert prep.provenance.marks.tolist().count(ENRICHED) == 3
    examples = [i for i, text in enumerate(prep.texts) if text.startswith("Kaiser")]
    assert len(examples) == 2 and prep.provenance.validate[examples].all(), "they validate again"
    assert _unscored(prep) == []


def test_a_holdout_scores_only_the_labels_its_test_split_has_real_rows_of(tmp_path):
    """A label balancing lifted has one or two real rows, and the stratified split puts
    the only one into train. Scored anyway, it reads F1 0.0 and drags the macro average
    down -- a number about the split, not the model (review #1)."""
    rows = [r for r in _rows() if r[2] != "C"]
    rows += [["Kaiser Augustus Quelle", "rom", "C", "", "", ""]]
    rows += [[f"Erzeugt Antike Text {i}", f"antike{i}", "C", "C", "", ""] for i in range(12)]

    prep = _prepare(tmp_path, _write(tmp_path, rows), cv_folds=0, stratified=True)

    in_test = prep.y_all[prep.test_idx].sum(axis=0) > 0
    assert "C" in _unscored(prep)
    assert _unscored(prep) == [c for c, hit in zip(prep.classes, in_test, strict=True) if not hit]


def test_a_holdout_that_loses_its_real_rows_to_the_label_drop_falls_back(tmp_path, monkeypatch):
    """Val and test drawn from real rows whose labels have no training positive: the
    unlearnable-label drop empties them. That is the same shortage as too few real rows,
    and gets the same fallback instead of a failed run (review #2)."""
    from app import data

    rows = [[f"Erzeugt Mathe Text {i}", f"zahl{i}", "A", "A", "", ""] for i in range(12)]
    rows += [[f"Photosynthese Versuch {i}", f"licht{i}", "B", "", "", ""] for i in range(3)]
    real_split = data.three_way_split

    def split(n, **kw):
        if kw.get("train_only") is None:
            return real_split(n, **kw)
        train_only = kw["train_only"]
        real = np.flatnonzero(~train_only)
        return np.flatnonzero(train_only), real[:1], real[1:]

    monkeypatch.setattr(data, "three_way_split", split)

    prep = _prepare(tmp_path, _write(tmp_path, rows), cv_folds=0, stratified=True)

    assert prep.provenance.fallback is not None
    assert prep.provenance.validate is None
    assert len(prep.val_idx) and len(prep.test_idx)


def test_excluding_every_row_says_that_rows_were_left_out(tmp_path):
    from app.errors import TrainingInputError

    generated = [[f"Erzeugt {lab} Text {i}", f"kw{i}", lab, lab, "", ""]
                 for lab in ("A", "B") for i in range(12)]

    with pytest.raises(TrainingInputError, match="24 generated rows were left out"):
        _prepare(tmp_path, _write(tmp_path, generated), cv_folds=3, synthetic_rows="exclude")
