"""Training end to end on a dataset whose rows an LLM partly wrote."""

import pytest

from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

HEADER = "title;keywords;subject;generated_for;example_for;enriched_fields"


def _rows() -> list[str]:
    rows = [f"Bruchrechnung Aufgabe {i};brüche zahlen {i};A;;;" for i in range(14)]
    rows += [f"Photosynthese Versuch {i};licht pflanze {i};B;;;{'keywords' if i < 3 else ''}"
             for i in range(14)]
    # Label C: two real rows, both shown to the generator, and twelve it wrote.
    rows += [f"Kaiser Augustus Quelle {i};rom antike {i};C;;C;" for i in range(2)]
    rows += [f"Erzeugt Antike Text {i};rom kaiser {i};C;C;;" for i in range(12)]
    return rows


def _train(tmp_path, lines: list[str], *, name: str, cv_folds: int, **req) -> tuple[dict, Registry]:
    (tmp_path / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    config = TrainingConfig(
        default_profile="fast", profiles={"fast": Profile("fast", "TF-IDF", True, True, [1.0, 2.0])},
        validation_size=0.2, test_size=0.2, min_text_length=5, drop_duplicates=True,
        min_samples_per_label=2,
    )
    registry = Registry(settings.models_dir, 4)
    request = {"dataset_name": f"{name}.csv", "model_name": name,
               "text_columns": ["title", "keywords"], "label_column": "subject",
               "csv_separator": ";", "label_separator": ",", "label_filter": None,
               "cv_folds": cv_folds, **req}
    result = run_training(request, settings, config, config.get("fast"), registry,
                          on_progress=lambda **_: None, should_stop=lambda: False)
    return result, registry


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_a_label_only_ai_rows_carry_is_trained_but_not_scored(tmp_path, cv_folds):
    result, registry = _train(tmp_path, [HEADER, *_rows()], name="marked", cv_folds=cv_folds)

    assert result["n_labels"] == 3, "C is trained: its rows count towards the minimum"
    assert set(result["metrics"]["per_label_f1"]) == {"A", "B"}
    assert "C" in registry.get("marked").classes


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_blank_mark_columns_train_the_model_an_unmarked_dataset_trains(tmp_path, cv_folds):
    """Backward compatibility, end to end: the same C, thresholds and metrics."""
    real = [row for row in _rows() if ";C;" not in row or "Kaiser" in row]
    plain = ["title;keywords;subject", *[";".join(r.split(";")[:3]) for r in real]]
    blank = [HEADER, *[";".join([*r.split(";")[:3], "", "", ""]) for r in real]]

    first, registry = _train(tmp_path, plain, name="plain", cv_folds=cv_folds)
    second, _ = _train(tmp_path, blank, name="blank", cv_folds=cv_folds)

    assert first["metrics"] == second["metrics"]
    a, b = registry.get("plain"), Registry(tmp_path / "models", 4).get("blank")
    assert (a.global_threshold, a.per_label_thresholds) == (b.global_threshold, b.per_label_thresholds)


@pytest.mark.parametrize("cv_folds", [0, 3])
def test_the_bundle_says_what_was_trained_on_and_what_was_scored(tmp_path, cv_folds):
    _, registry = _train(tmp_path, [HEADER, *_rows()], name="marked", cv_folds=cv_folds)

    meta = registry.info("marked")["metadata"]
    block = meta["synthetic_data"]
    assert block["mode"] == "train"
    assert (block["generated_rows"], block["example_rows"], block["enriched_rows"]) == (12, 2, 3)
    assert block["train_only_rows"] == 17
    assert block["validated_on"] == "real_rows"
    assert block["labels_not_validated"] == ["C"]
    assert block["scored_rows"] == (25 if cv_folds else meta["n_test"])
    # A fixture this small IS a run whose numbers do not compare: 25 real rows out of
    # fold, or a 6-row test split. The k-fold arm clears the bar, the holdout arm does not.
    assert (block["too_few_rows"] is None) is bool(cv_folds)
    assert "AI-marked" in meta["evaluation"]


def test_a_bundle_whose_metrics_rest_on_a_few_rows_says_so(tmp_path):
    """End to end: the shortage has to survive prepare -> provenance -> the bundle.

    Eight real rows behind forty an LLM wrote. Validation and test are drawn from the
    eight, so two rows carry the metrics and two tune the thresholds — which the old code
    accepted as a real holdout split, reported as `validated_on: real_rows`, and said
    nothing further about. The fallback never fired because neither part was empty.
    """
    rows = [f"Bruchrechnung Aufgabe {i};brüche zahlen {i};A;;;" for i in range(4)]
    rows += [f"Wiener Kongress Quelle {i};geschichte europa {i};B;;;" for i in range(4)]
    rows += [f"Erzeugt {lab} Text {i};stichwort {lab} {i};{lab};{lab};;"
             for lab in ("A", "B") for i in range(20)]

    _, registry = _train(tmp_path, [HEADER, *rows], name="fewreal", cv_folds=0)

    block = registry.info("fewreal")["metadata"]["synthetic_data"]
    assert block["validated_on"] == "real_rows", "the real rows stay the ones measured on"
    assert block["fallback"] is None, "not converted into the all-rows fallback"
    assert block["scored_rows"] == 2
    assert block["too_few_rows"] == ("the metrics rest on 2 rows, fewer than the 10 a "
                                     "number comparable with another run needs")


def test_leaving_the_generated_rows_out_is_recorded_too(tmp_path):
    _, registry = _train(tmp_path, [HEADER, *_rows()], name="without", cv_folds=3,
                         synthetic_rows="exclude")

    block = registry.info("without")["metadata"]["synthetic_data"]
    assert (block["mode"], block["excluded_generated_rows"], block["generated_rows"]) == ("exclude", 12, 0)
    assert block["labels_not_validated"] == []


def test_a_dataset_with_nothing_marked_has_no_synthetic_block(tmp_path):
    real = ["title;keywords;subject", *[";".join(r.split(";")[:3]) for r in _rows()[:28]]]

    _, registry = _train(tmp_path, real, name="plain", cv_folds=3)

    meta = registry.info("plain")["metadata"]
    assert "synthetic_data" not in meta
    assert "AI-marked" not in meta["evaluation"]


def test_a_pure_ai_dataset_trains_and_its_bundle_says_the_numbers_are_not_real(tmp_path):
    generated = [f"Erzeugt {lab} Text {i};stichwort {lab} {i};{lab};{lab};;"
                 for lab in ("A", "B") for i in range(14)]

    _, registry = _train(tmp_path, [HEADER, *generated], name="pure", cv_folds=3)

    block = registry.info("pure")["metadata"]["synthetic_data"]
    assert block["validated_on"] == "all_rows"
    assert block["fallback"]



@pytest.mark.parametrize("mode", ["own", "global"])
def test_the_bundle_names_its_thin_labels_and_the_cut_they_got(tmp_path, mode):
    """C reached the minimum only through AI-marked rows. Whether it keeps a cut tuned
    on its real rows or takes the global one is the run's choice, and the bundle says
    which."""
    _, registry = _train(tmp_path, [HEADER, *_rows()], name="thin", cv_folds=3,
                         thin_label_threshold=mode)

    block = registry.info("thin")["metadata"]["synthetic_data"]
    assert block["thin_labels"] == ["C"]
    assert block["thin_label_threshold"] == mode
    model = registry.get("thin")
    if mode == "global":
        assert model.per_label_thresholds.get("C", model.global_threshold) == model.global_threshold
