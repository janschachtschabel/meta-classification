"""Tests for dataset loading, cleaning and label preparation."""

from pathlib import Path

from app import data

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.csv"
TEXT_COLS = ["properties.cclom:title", "properties.cclom:general_keyword"]
LABEL_COL = "properties.ccm:taxonid"


def test_clean_text():
    assert data.clean_text("  a&amp;b\t c ") == "a&b c"
    assert data.clean_text(None) == ""
    assert data.clean_text("x\x00y") == "xy"
    # HTML tags and Markdown markup are stripped.
    assert data.clean_text("<p>Hallo <b>Welt</b></p>") == "Hallo Welt"
    assert data.clean_text("# Titel **fett** [link](http://x)") == "Titel fett link"


def test_split_labels():
    assert data.split_labels("a, b ,,c") == ["a", "b", "c"]
    assert data.split_labels("") == []
    assert data.split_labels(None) == []


def test_count_rows_cached_by_mtime_and_size(tmp_path, monkeypatch):
    """Row counts are served from a (mtime, size) cache: repeated dataset
    listings must not re-read unchanged multi-MB CSVs line by line."""
    csv = tmp_path / "t.csv"
    csv.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    assert data.count_rows(csv) == 2

    def boom(*args, **kwargs):
        raise AssertionError("file re-read although it is unchanged (cache miss)")

    monkeypatch.setattr("builtins.open", boom)
    assert data.count_rows(csv) == 2  # unchanged -> cache hit, no open()
    monkeypatch.undo()

    csv.write_text("a;b\n1;2\n3;4\n5;6\n", encoding="utf-8")
    assert data.count_rows(csv) == 3  # size/mtime changed -> recount


def test_detect_task_type():
    assert data.detect_task_type([["a"], ["b"], ["a"]], 3) == "multiclass"
    assert data.detect_task_type([["a"], ["b"]], 2) == "binary"
    assert data.detect_task_type([["a", "b"], ["a"]], 2) == "multilabel"


def test_prepare_targets_drops_rare_labels_and_empty_rows():
    labels = [["a"], ["a"], ["a"], ["b"]]  # 'b' occurs only once
    matrix, classes, row_keep = data.prepare_targets(labels, min_samples=2)
    assert classes == ["a"]
    assert matrix.shape == (3, 1)
    assert row_keep.tolist() == [True, True, True, False]


def test_auto_min_samples_scales():
    assert data.auto_min_samples(500) == 2
    assert data.auto_min_samples(5_000) == 5
    assert data.auto_min_samples(100_000) == 35
    assert data.auto_min_samples(100_000, override=3) == 3


def test_load_dataset_falls_back_to_cp1252(tmp_path):
    """A legitimate non-UTF-8 CSV (Windows-1252, common for German exports) must
    load via the fallback instead of raising an opaque UnicodeDecodeError."""
    csv = tmp_path / "de.csv"
    content = (
        "properties.cclom:title;properties.cclom:general_keyword;properties.ccm:taxonid\n"
        "Brüche üben und Größen;Mathematik;math\n"
    )
    csv.write_bytes(content.encode("cp1252"))  # 'ü'/'ö' -> 0xFC/0xF6, invalid UTF-8
    loaded = data.load_dataset(csv, TEXT_COLS, LABEL_COL)
    assert loaded.texts and "Brüche" in loaded.texts[0]


def test_load_dataset_maps_empty_csv_to_training_input_error(tmp_path):
    from app.errors import TrainingInputError

    csv = tmp_path / "empty.csv"
    csv.write_bytes(b"")
    try:
        data.load_dataset(csv, TEXT_COLS, LABEL_COL)
    except TrainingInputError:
        return
    raise AssertionError("empty CSV should raise TrainingInputError, not a raw pandas error")


def test_load_dataset_reads_fixture():
    loaded = data.load_dataset(FIXTURE, TEXT_COLS, LABEL_COL)
    assert len(loaded.texts) == 36
    assert len(loaded.label_lists) == 36
    assert loaded.uri_to_label["uri:math"] == "Mathematik"
    assert all(len(labs) == 1 for labs in loaded.label_lists)


def test_three_way_split_is_disjoint_and_complete():
    train, val, test = data.three_way_split(100, val_size=0.15, test_size=0.15, seed=42)
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
