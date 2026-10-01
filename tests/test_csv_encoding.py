"""A dataset's encoding is decided once, for the whole file, from its bytes (audit 2026-09-30, T02).

Every reader used to decide by trial: UTF-8, and on the first undecodable byte anywhere, the
whole file again as cp1252. One truncated umlaut in a clean UTF-8 export turned every "ä" of
every row into "Ã¤"; training and test saw the same garbage, so the metrics did not show it,
and the model predicted "math" at 0.37 for texts the clean file's model got right at 0.9.
"""

import gzip
from pathlib import Path

import pytest

from app import csv_encoding
from app.dataset_load import load_dataset
from app.dataset_stats import sample_rows
from app.errors import TrainingInputError

HEADER = "title;label\n"
CLEAN = [f"Übungen zur Bruchrechnung für Klasse {i} mit Größen;uri:math\n" for i in range(60)]
CLEAN += [f"Französische Revolution und ihre Folgen, Quelle {i};uri:hist\n" for i in range(60)]
CLEAN += [f"Photosynthese in grünen Blättern, Versuch {i};uri:bio\n" for i in range(60)]


def _write(tmp_path: Path, payload: bytes, name: str = "set.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(gzip.compress(payload) if name.endswith(".gz") else payload)
    return path


def _utf8(*extra: bytes) -> bytes:
    return (HEADER + "".join(CLEAN)).encode("utf-8") + b"".join(extra)


def test_a_clean_utf8_file_is_utf8(tmp_path):
    assert csv_encoding.detect(_write(tmp_path, _utf8())) == csv_encoding.CsvEncoding("utf-8")


def test_a_cp1252_file_is_cp1252(tmp_path):
    payload = (HEADER + "".join(CLEAN)).encode("cp1252")

    assert csv_encoding.detect(_write(tmp_path, payload)) == csv_encoding.CsvEncoding("cp1252")


def test_a_cp1252_file_stays_cp1252_where_its_bytes_form_a_utf8_pair_by_accident(tmp_path):
    """`heiß“` is DF 93 in cp1252 -- a valid two-byte UTF-8 sequence. German exports have it."""
    payload = (HEADER + "".join(CLEAN) + "Das Wort „heiß“ im Satz;uri:de\n").encode("cp1252")

    assert csv_encoding.detect(_write(tmp_path, payload)).name == "cp1252"


def test_one_truncated_umlaut_leaves_a_utf8_file_utf8(tmp_path):
    """The audit's case: `0xC3` whose second byte was cut off, in an otherwise clean export."""
    path = _write(tmp_path, _utf8(b"Abgeschnittenes \xc3 am Ende;uri:math\n"))

    assert csv_encoding.detect(path) == csv_encoding.CsvEncoding("utf-8", replaced=1)


def test_a_typographic_quote_does_not_make_the_file_unreadable(tmp_path):
    """E2 80 9D is `”` in UTF-8 and holds 0x9D, a byte cp1252 does not define: read as
    cp1252, the whole training failed with "see server logs"."""
    path = _write(tmp_path, _utf8(b"Zitat \xe2\x80\x9dso\xe2\x80\x9d und \xc3 kaputt;uri:math\n"))

    assert csv_encoding.detect(path) == csv_encoding.CsvEncoding("utf-8", replaced=1)


def test_more_than_a_handful_of_damaged_bytes_is_refused_with_where(tmp_path):
    damage = b"".join(b"kaputt \xc3 Zeile;uri:math\n" for _ in range(csv_encoding.MAX_REPLACED + 1))
    clean = _utf8()
    path = _write(tmp_path, clean + damage)

    with pytest.raises(TrainingInputError, match=f"byte {len(clean) + 7}"):
        csv_encoding.detect(path)


def test_a_file_mixing_both_encodings_is_refused(tmp_path):
    payload = _utf8() + "".join(CLEAN).encode("cp1252")

    with pytest.raises(TrainingInputError, match="UTF-8"):
        csv_encoding.detect(_write(tmp_path, payload))


def test_a_byte_neither_encoding_defines_is_refused_with_where(tmp_path):
    payload = (HEADER + "Steuerzeichen \x9d mitten drin;uri:math\n").encode("latin-1")

    with pytest.raises(TrainingInputError, match="byte 26"):
        csv_encoding.detect(_write(tmp_path, payload))


def test_a_character_split_across_two_blocks_is_not_damage(tmp_path):
    path = _write(tmp_path, (HEADER + CLEAN[0] + "Straße, Größe, Äpfel, „Zitat“ — €;x\n").encode("utf-8"))
    for block in range(1, 41):  # every cut position through two- and three-byte characters
        assert csv_encoding.detect(path, block_bytes=block) == csv_encoding.CsvEncoding("utf-8")


def test_a_gzipped_file_is_judged_by_its_content(tmp_path):
    path = _write(tmp_path, _utf8(b"Abgeschnittenes \xc3 am Ende;uri:math\n"), name="set.csv.gz")

    assert csv_encoding.detect(path) == csv_encoding.CsvEncoding("utf-8", replaced=1)


def test_training_reads_the_damaged_utf8_file_as_utf8(tmp_path):
    path = _write(tmp_path, _utf8(b"Abgeschnittenes \xc3 am Ende;uri:math\n"))

    loaded = load_dataset(path, ["title"], "label", separator=";")

    assert any("Übungen zur Bruchrechnung" in text for text in loaded.texts)
    assert not any("Ã" in text for text in loaded.texts), "the file was read as cp1252"
    assert (loaded.encoding.name, loaded.encoding.replaced) == ("utf-8", 1)


def test_training_refuses_a_mixed_file_with_a_reason(tmp_path):
    path = _write(tmp_path, _utf8() + "".join(CLEAN).encode("cp1252"))

    with pytest.raises(TrainingInputError, match="byte"):
        load_dataset(path, ["title"], "label", separator=";")


def test_the_dataset_preview_reads_the_damaged_file_as_utf8(tmp_path):
    path = _write(tmp_path, HEADER.encode() + b"Abgeschnitten \xc3 hier;uri:math\n" + _utf8()[len(HEADER):])

    preview = sample_rows(path, separator=";", n=3)

    assert preview["sample"][1]["title"].startswith("Übungen")


def test_classifying_a_csv_reads_the_damaged_file_as_utf8(tmp_path):
    """The same rule for /predict/csv: its stream cannot change its mind after byte one."""
    from app import predict_csv

    path = _write(tmp_path, _utf8(b"Abgeschnittenes \xc3 am Ende;uri:math\n"))
    seen: list[str] = []

    class Recorder:
        def predict(self, texts, **_kwargs):
            seen.extend(texts)
            return [[] for _ in texts]

    "".join(predict_csv.classify_csv(path, Recorder(), text_columns=["title"], separator=";"))

    assert seen[0].startswith("Übungen zur Bruchrechnung"), seen[0]
    assert predict_csv.check_input(path, ["title"], separator=";").encoding == csv_encoding.CsvEncoding(
        "utf-8", replaced=1), "the check decides the encoding the stream is read in"


def test_the_bundle_records_how_its_dataset_was_decoded(tmp_path):
    from app.profiles import Profile, TrainingConfig
    from app.registry import Registry
    from app.settings import Settings
    from app.training import run_training

    _write(tmp_path, _utf8(b"Abgeschnittenes \xc3 am Ende;uri:math\n"))
    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    config = TrainingConfig(default_profile="fast",
                            profiles={"fast": Profile("fast", "TF-IDF", True, True, [1.0])},
                            validation_size=0.2, test_size=0.2, min_text_length=5,
                            drop_duplicates=True, min_samples_per_label=2)
    request = {"dataset_name": "set.csv", "model_name": "m", "text_columns": ["title"],
               "label_column": "label", "csv_separator": ";", "label_separator": ",",
               "label_filter": None}

    run_training(request, settings, config, config.get("fast"), Registry(settings.models_dir, 2),
                 on_progress=lambda **_: None, should_stop=lambda: False)

    _model, metadata = Registry(settings.models_dir, 2).load_fresh("m")
    assert metadata["csv_encoding"] == {"name": "utf-8", "replaced_bytes": 1}
