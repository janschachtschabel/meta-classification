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
    sits blocks into the file, the rows already read as UTF-8 must be thrown away and
    the file read again — not kept, and not read twice into the result."""
    # Plain ASCII up to the bad row — it reads the same in both encodings — so the UTF-8
    # attempt succeeds for 39 blocks before it meets the first cp1252 byte.
    rows = [f'"Eintrag {i} zu Themen";"uri:topic{i % 5}"\n' for i in range(40_000)]
    rows[39_000] = '"Die Größe der Flächen";"uri:math"\n'
    path = tmp_path / "export.csv"
    path.write_bytes(("title;labels\n" + "".join(rows)).encode("cp1252"))

    reader = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", chunksize=1000)
    read_before_failing = 0
    with pytest.raises(UnicodeDecodeError), reader:
        for _chunk in reader:
            read_before_failing += 1
    assert read_before_failing >= 1, "premise: UTF-8 must fail only after a first block"

    # drop_duplicates off: with it on, rows read twice would vanish as duplicates and a
    # loader that kept the UTF-8 attempt's rows would still look right.
    loaded = load_dataset(path, ["title"], "labels", chunk_rows=1000, drop_duplicates=False)
    assert len(loaded.texts) == 40_000, "rows read before the restart must not stay"
    assert loaded.texts[0] == "Eintrag 0 zu Themen"
    assert loaded.texts[39_000] == "Die Größe der Flächen"
