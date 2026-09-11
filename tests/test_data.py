"""Tests for dataset loading, cleaning and label preparation."""

from pathlib import Path

from app import data, dataset_load

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


def test_prepare_targets_uses_a_compact_dtype():
    """The target matrix holds 0/1 and is the second-largest array in a training run
    (rows x labels, densely). int64 spends 8 bytes per bit: at 600k rows x 300 labels
    that is 1.34 GB instead of 168 MB. Values and comparisons are unaffected."""
    import numpy as np

    matrix, _classes, _keep = data.prepare_targets([["a"], ["a"], ["b"], ["b"]], min_samples=2)
    assert matrix.dtype == np.int8
    assert matrix.sum() == 4  # arithmetic still behaves
    assert set(np.unique(matrix)) <= {0, 1}


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
    loaded = dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL)
    assert loaded.texts and "Brüche" in loaded.texts[0]


def test_load_dataset_maps_empty_csv_to_training_input_error(tmp_path):
    from app.errors import TrainingInputError

    csv = tmp_path / "empty.csv"
    csv.write_bytes(b"")
    try:
        dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL)
    except TrainingInputError:
        return
    raise AssertionError("empty CSV should raise TrainingInputError, not a raw pandas error")


def test_load_dataset_reads_fixture():
    loaded = dataset_load.load_dataset(FIXTURE, TEXT_COLS, LABEL_COL)
    assert len(loaded.texts) == 36
    assert len(loaded.label_lists) == 36
    assert loaded.uri_to_label["uri:math"] == "Mathematik"
    assert all(len(labs) == 1 for labs in loaded.label_lists)


def _displayname_csv(tmp_path, rows: list[tuple[str, str]]) -> Path:
    """A CSV with a label column and its _DISPLAYNAME twin, both comma-separated."""
    csv = tmp_path / "dn.csv"
    lines = [f"{LABEL_COL};{LABEL_COL}_DISPLAYNAME;{TEXT_COLS[0]};{TEXT_COLS[1]}"]
    lines += [f"{uris};{names};some title text;keyword" for uris, names in rows]
    csv.write_text("\n".join(lines), encoding="utf-8")
    return csv


def test_count_rows_reads_gzipped_csv(tmp_path):
    """A .csv.gz must be counted decompressed, not as raw bytes.

    pandas already reads gzip transparently, so training on a compressed dataset worked —
    but `count_rows` opened the file in text mode and counted the *compressed* bytes'
    newlines. Measured on the 426,724-row WLO export, that reported 1,030,307. Wrong by more
    than a factor of two, and shown in every dataset listing.
    """
    import gzip

    body = "a;b\n" + "".join(f"{i};{i}\n" for i in range(500))
    plain = tmp_path / "plain.csv"
    plain.write_text(body, encoding="utf-8")
    packed = tmp_path / "packed.csv.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as handle:
        handle.write(body)

    assert data.count_rows(plain) == 500
    assert data.count_rows(packed) == 500


def test_count_rows_ignores_newlines_inside_quoted_fields(tmp_path):
    """A line break inside a quoted description is not a new record.

    Counting physical lines was documented as a harmless approximation, but on real WLO data
    it is not: `data_300k.csv` reported 1,343,683 rows for 340,630 records (3.94x), the
    combined WLO export 2,141,123 for 426,724 (5.02x). Descriptions simply contain many
    newlines. A number that wrong in the dataset listing is misinformation, not a shortcut.
    """
    csv = tmp_path / "quoted.csv"
    csv.write_text(
        'title;description\n'
        'A;"line one\nline two\nline three"\n'
        'B;"und ein ""Zitat"" mit\nUmbruch"\n'
        'C;plain\n',
        encoding="utf-8",
    )
    assert data.count_rows(csv) == 3


def test_container_labels_never_become_trainable_classes(tmp_path):
    """A value naming a NAMESPACE rather than a concept must not train as a label.

    Real case: 522 rows in data_300k.csv were tagged with the bare vocabulary root
    `http://w3id.org/openeduhub/vocabs/discipline/` — no concept id at all. It trained as
    an ordinary class scoring F1 0.4096, diluting macro F1 and letting `/predict` return a
    label that means nothing. `min_samples_per_label` cannot catch it: 522 rows clears any
    sane threshold, so the guard has to be structural.

    A trailing separator marks a container in every hierarchical identifier scheme (URI,
    path), so this needs no vocabulary knowledge.
    """
    csv = tmp_path / "c.csv"
    csv.write_text(
        f"{LABEL_COL};{TEXT_COLS[0]};{TEXT_COLS[1]}\n"
        "http://x/vocabs/discipline/,http://x/vocabs/discipline/120;title one;keyword\n"
        "http://x/vocabs/discipline/;title two;keyword\n"
        "http://x/vocabs/discipline/380;title three;keyword\n",
        encoding="utf-8",
    )
    loaded = dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL, min_text_length=0)

    trained = {label for labs in loaded.label_lists for label in labs}
    assert trained == {"http://x/vocabs/discipline/120", "http://x/vocabs/discipline/380"}
    # Row 2 held ONLY the container, so it carries no label at all and must be dropped
    # rather than kept as an all-zero target (same contract as label_filter).
    assert len(loaded.texts) == 2


def test_label_hierarchy_levels_are_kept_only_containers_go(tmp_path):
    """Guard against over-reach: a legitimate broader concept has an id and must survive.

    The higher-education vocabulary is genuinely hierarchical — a row carries Fächergruppe
    AND Studienbereich AND Fach, and 2.25 labels per row is exactly that redundancy the
    model learns from. Only the id-less root is invalid, never a level of the hierarchy.
    """
    csv = tmp_path / "h.csv"
    csv.write_text(
        f"{LABEL_COL};{TEXT_COLS[0]};{TEXT_COLS[1]}\n"
        "http://x/v/n1,http://x/v/n09,http://x/v/;title;keyword\n",
        encoding="utf-8",
    )
    loaded = dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL, min_text_length=0)
    assert loaded.label_lists == [["http://x/v/n1", "http://x/v/n09"]]


def test_display_names_are_never_misattributed_when_a_name_contains_a_comma(tmp_path):
    """A comma INSIDE a display name must not shift every later name onto the wrong URI.

    Both columns are comma-separated and were zipped positionally, so
    "Germanistik (Deutsch, germanische Sprachen)" split into two entries and every pair
    after it slid by one — measured on data_300k.csv, that mislabelled 34 of 119 higher-ed
    labels with the name of a DIFFERENT subject. Showing the wrong subject is worse than
    showing none, so an unattributable row must contribute nothing.
    """
    csv = _displayname_csv(tmp_path, [
        ("uri:a,uri:b", "Physik,Chemie"),  # aligned -> trustworthy
        ("uri:c,uri:d", "Germanistik (Deutsch, germanische Sprachen),Sport"),
    ])
    loaded = dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL, min_text_length=0)

    assert loaded.uri_to_label["uri:a"] == "Physik"
    assert loaded.uri_to_label["uri:b"] == "Chemie"
    # The row above splits into 3 fragments for 2 URIs. Re-joining the lowercase
    # continuation lands back on 2, so both names are recoverable — and crucially
    # `uri:d` must NEVER become "Germanistik (Deutsch".
    assert loaded.uri_to_label["uri:c"] == "Germanistik (Deutsch, germanische Sprachen)"
    assert loaded.uri_to_label["uri:d"] == "Sport"


def test_unattributable_display_names_are_dropped_rather_than_guessed(tmp_path):
    """When the fragment count cannot be reconciled with the URI count, emit no name.

    The URI count is authoritative (URIs contain no commas). If re-joining still does not
    reach it, the row is ambiguous and any pairing would be a guess. `/predict` falls back
    to the URI, which is honest; a wrong subject name is not.
    """
    csv = _displayname_csv(tmp_path, [
        # Two comma-bearing names at once, both continuing with an uppercase word:
        # nothing in the text says where the boundary is.
        ("uri:x,uri:y", "Rechts, Wirtschaft, Soziales,Kunst, Musik"),
    ])
    loaded = dataset_load.load_dataset(csv, TEXT_COLS, LABEL_COL, min_text_length=0)
    assert "uri:x" not in loaded.uri_to_label
    assert "uri:y" not in loaded.uri_to_label


def test_authoritative_label_names_override_the_csv(tmp_path):
    """An external vocabulary (SKOS) wins over names derived from the CSV.

    `scripts/fetch_vocab_labels.py` writes such a file; it is the only complete source,
    because a comma-corrupted export cannot be fully repaired from the export alone.
    """
    csv = _displayname_csv(tmp_path, [("uri:a,uri:b", "Falsch,Auch falsch")])
    loaded = dataset_load.load_dataset(
        csv, TEXT_COLS, LABEL_COL, min_text_length=0,
        label_names={"uri:a": "Physik", "uri:c": "Nicht im Datensatz"},
    )
    assert loaded.uri_to_label["uri:a"] == "Physik"       # authoritative wins
    assert loaded.uri_to_label["uri:b"] == "Auch falsch"  # no authority -> keep the CSV name
    assert "uri:c" not in loaded.uri_to_label             # unused labels are not carried along


def test_text_column_weights_repeat_a_field_in_the_combined_text(tmp_path):
    """A weight of N repeats that column N times when the training text is built, so
    a short, dense field (title, keywords) is not drowned out by a long description.
    Columns without a weight stay at 1x, and column ORDER is preserved."""
    csv = tmp_path / "w.csv"
    csv.write_text(
        "properties.cclom:title;properties.cclom:general_description;properties.ccm:taxonid\n"
        "Bruch;lange Beschreibung;math\n",
        encoding="utf-8",
    )
    cols = ["properties.cclom:title", "properties.cclom:general_description"]

    plain = dataset_load.load_dataset(csv, cols, LABEL_COL)
    assert plain.texts == ["Bruch lange Beschreibung"]

    weighted = dataset_load.load_dataset(
        csv, cols, LABEL_COL, text_column_weights={"properties.cclom:title": 3}
    )
    assert weighted.texts == ["Bruch Bruch Bruch lange Beschreibung"]


def test_text_column_weights_ignore_columns_absent_from_the_csv(tmp_path):
    """load_dataset already skips requested columns the CSV lacks; a weight for such
    a column must not resurrect it or shift the others."""
    csv = tmp_path / "w2.csv"
    csv.write_text(
        "properties.cclom:title;properties.ccm:taxonid\nBruch;math\n", encoding="utf-8"
    )
    loaded = dataset_load.load_dataset(
        csv,
        ["properties.cclom:title", "properties.cclom:general_description"],
        LABEL_COL,
        text_column_weights={"properties.cclom:general_description": 5},
    )
    assert loaded.texts == ["Bruch"]


def test_validate_dataset_warns_about_rows_without_labels(tmp_path):
    """The validate endpoint promises a 'rows without labels' warning; the loader
    silently drops such rows, so the count must come from the raw label column."""
    from app import dataset_stats

    csv = tmp_path / "d.csv"
    csv.write_text(
        "properties.cclom:title;properties.ccm:taxonid\n"
        "Bruchrechnung üben;math\n"
        "Text ohne Label;\n"
        "Noch einer ohne;\n",
        encoding="utf-8",
    )
    result = dataset_stats.validate_dataset(csv, ["properties.cclom:title"], "properties.ccm:taxonid")
    assert result["valid"] is True
    assert any("2 rows without labels" in w for w in result["warnings"]), result["warnings"]


def test_three_way_split_is_disjoint_and_complete():
    train, val, test = data.three_way_split(100, val_size=0.15, test_size=0.15, seed=42)
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
