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
