"""The loader reads a CSV in blocks of rows, and nothing it returns may notice.

Reading the whole file held three full-length copies of the text at once — the frame,
the combined series and the cleaned one; +1.5 GB for a 558 MB file
(docs/plans/2026-09-11-training-memory.md). In blocks, each copy is one block long. What
must not change: row order, which duplicate survives, which display name a URI keeps,
the authoritative names, and the UTF-8 -> cp1252 fallback. The expected values below
are what the whole-file loader returned for this CSV before it read in blocks.
"""

import gzip

import pandas as pd
import pytest

from app.dataset_load import load_dataset
from app.errors import TrainingInputError

ROWS = [
    ("Bruchrechnung", "Brüche <b>kürzen</b>", "uri:math", "Mathematik"),
    ("Rom", "", "uri:hist", "Geschichte"),
    ("Photosynthese", "Pflanzen & Licht", "", ""),
    ("Das Römische Reich", "Die Antike", "uri:hist,uri:ancient", "Geschichte,Antike"),
    ("Zellbiologie", "Die **Zelle**", "uri:bio", ""),
    ("Gleichungen", "lösen üben", "uri:math", "Mathe"),
    ("Säuren", "und Basen", "uri:chem,uri:bio", "Chemie,Biologie"),
    ("Bruchrechnung", "Brüche <b>kürzen</b>", "uri:math", "Mathematik"),
    ("Gedichte", "der Romantik", "uri:german", "Deutsch"),
    ("Kaiser Augustus", "Rom", "uri:hist", "Historie"),
    ("Zellbiologie", "Die **Zelle**", "uri:bio", "Biologie"),
    ("Grammatik", "Kommaregeln", "uri:german", ""),
]
CSV = "title;description;labels;labels_DISPLAYNAME\n" + "".join(
    ";".join(f'"{cell}"' for cell in row) + "\n" for row in ROWS)
NAMES = [("uri:math", "Mathematik"), ("uri:hist", "Geschichte"), ("uri:ancient", "Antike"),
         ("uri:chem", "Chemie"), ("uri:bio", "Biologie"), ("uri:german", "Deutsch")]

CASES = {
    "defaults": ({}, dict(
        texts=["Bruchrechnung Brüche kürzen", "Das Römische Reich Die Antike",
               "Zellbiologie Die Zelle", "Gleichungen lösen üben", "Säuren und Basen",
               "Gedichte der Romantik", "Kaiser Augustus Rom", "Grammatik Kommaregeln"],
        labels=[["uri:math"], ["uri:hist", "uri:ancient"], ["uri:bio"], ["uri:math"],
                ["uri:chem", "uri:bio"], ["uri:german"], ["uri:hist"], ["uri:german"]],
        names=NAMES,
    )),
    "weighted+names": (dict(
        text_column_weights={"title": 2},
        label_names={"uri:bio": "Biologie (Vokabular)", "uri:physics": "Physik"},
    ), dict(
        texts=["Bruchrechnung Bruchrechnung Brüche kürzen", "Rom Rom",
               "Das Römische Reich Das Römische Reich Die Antike",
               "Zellbiologie Zellbiologie Die Zelle", "Gleichungen Gleichungen lösen üben",
               "Säuren Säuren und Basen", "Gedichte Gedichte der Romantik",
               "Kaiser Augustus Kaiser Augustus Rom", "Grammatik Grammatik Kommaregeln"],
        labels=[["uri:math"], ["uri:hist"], ["uri:hist", "uri:ancient"], ["uri:bio"],
                ["uri:math"], ["uri:chem", "uri:bio"], ["uri:german"], ["uri:hist"],
                ["uri:german"]],
        names=[*NAMES[:4], ("uri:bio", "Biologie (Vokabular)"), NAMES[5]],
    )),
    "filter+keep-dups": (dict(label_filter="uri:h", drop_duplicates=False, min_text_length=0),
                         dict(texts=["Rom", "Das Römische Reich Die Antike", "Kaiser Augustus Rom"],
                              labels=[["uri:hist"], ["uri:hist"], ["uri:hist"]],
                              names=NAMES)),
    # The authoritative names apply to every label the file USES, filtered away or not:
    # neither uri:ancient nor uri:chem survives the filter, and both still get renamed.
    "filter+names": (dict(label_filter="uri:h",
                          label_names={"uri:ancient": "Antike (V)", "uri:chem": "Chemie (V)"}),
                     dict(texts=["Das Römische Reich Die Antike", "Kaiser Augustus Rom"],
                          labels=[["uri:hist"], ["uri:hist"]],
                          names=[*NAMES[:2], ("uri:ancient", "Antike (V)"),
                                 ("uri:chem", "Chemie (V)"), *NAMES[4:]])),
}


def _check(loaded, expected: dict) -> None:
    assert loaded.texts == expected["texts"]
    assert loaded.label_lists == expected["labels"]
    assert list(loaded.uri_to_label.items()) == expected["names"]  # order included


@pytest.mark.parametrize("chunk_rows", [1, 7, 10_000])
@pytest.mark.parametrize("case", list(CASES))
def test_reading_in_blocks_returns_what_the_whole_file_returned(tmp_path, case, chunk_rows):
    """Block size 1 puts every duplicate, every late display name and every row the
    authoritative names apply to on the far side of a block boundary."""
    path = tmp_path / "crafted.csv"
    path.write_text(CSV, encoding="utf-8")
    kwargs, expected = CASES[case]
    _check(load_dataset(path, ["title", "description"], "labels", chunk_rows=chunk_rows,
                        **kwargs), expected)


@pytest.mark.parametrize("chunk_rows", [1, 7, 10_000])
def test_a_gzipped_csv_reads_in_blocks_too(tmp_path, chunk_rows):
    path = tmp_path / "crafted.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(CSV)
    kwargs, expected = CASES["defaults"]
    _check(load_dataset(path, ["title", "description"], "labels", chunk_rows=chunk_rows,
                        **kwargs), expected)


def test_a_cp1252_byte_deep_in_the_file_restarts_the_whole_read(tmp_path):
    """German metadata exports are commonly Windows-1252. When the first non-UTF-8 byte
    sits blocks into the file, everything read as UTF-8 must be thrown away and the file
    read again from its start — not kept, not read twice into the result, and not resumed
    in the other encoding: the whole-file loader decoded ALL of such a file as cp1252."""
    # Row 0 holds what cp1252 writes for "Ã¤" (bytes C3 A4), which UTF-8 happily reads as
    # "ä" — the double-encoded text such exports carry. Only a read that started over
    # returns it as cp1252, in the text and in the display name. The other rows are ASCII,
    # the same in both encodings, so the UTF-8 attempt runs 39 blocks before it fails.
    rows = [f'"Eintrag {i} zu Themen";"uri:topic{i % 5}";"Thema {i % 5}"\n'
            for i in range(40_000)]
    rows[0] = '"Ã¤rger mit Umlauten";"uri:first";"Ã¤rger"\n'
    rows[39_000] = '"Die Größe der Flächen";"uri:math";"Mathematik"\n'
    path = tmp_path / "export.csv"
    path.write_bytes(("title;labels;labels_DISPLAYNAME\n" + "".join(rows)).encode("cp1252"))

    reader = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", chunksize=1000)
    first_titles = []
    with pytest.raises(UnicodeDecodeError), reader:
        for chunk in reader:
            first_titles.append(chunk["title"].iloc[0])
    assert first_titles, "premise: UTF-8 must fail only after a first block"
    assert first_titles[0] == "ärger mit Umlauten", "premise: UTF-8 reads row 0 differently"

    # drop_duplicates off: with it on, rows read twice would vanish as duplicates and a
    # loader that kept the UTF-8 attempt's rows would still look right.
    loaded = load_dataset(path, ["title"], "labels", chunk_rows=1000, drop_duplicates=False)
    assert len(loaded.texts) == 40_000, "rows read before the restart must not stay"
    assert loaded.texts[0] == "Ã¤rger mit Umlauten", "the first block, decoded again"
    assert loaded.uri_to_label["uri:first"] == "Ã¤rger", "its display name too"
    assert loaded.texts[39_000] == "Die Größe der Flächen"
    # And with it on: texts the UTF-8 attempt had already seen must not count as seen.
    assert len(load_dataset(path, ["title"], "labels", chunk_rows=1000).texts) == 40_000


def test_a_malformed_row_in_a_later_block_is_an_input_error(tmp_path):
    """The header parses and so do the first blocks; the quote that never closes sits in
    the last one. It must still reach the operator as the crafted 400, not a raw 500."""
    path = tmp_path / "broken.csv"
    # A quote that never closes — not '"offen;"uri:hist', which pandas reads leniently.
    path.write_text('title;labels\n"Bruchrechnung";"uri:math"\n"Gleichungen";"uri:math"\n'
                    '"Das Römische Reich";"uri:hist"\n"offen;uri:hist\n', encoding="utf-8")
    with pytest.raises(TrainingInputError, match="empty or malformed"):
        load_dataset(path, ["title"], "labels", chunk_rows=1)
