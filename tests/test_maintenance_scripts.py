"""The maintenance scripts under scripts/ that are not the bundle repairs (audit 2026-09-30,
W03, W05, W06, W07).

None of them runs inside the API, and each prepares or judges what the API trains on -- so a
defect there reaches a model without ever touching app/. They are loaded by path, as their
own `python scripts/<name>.py` would, and their network and model calls are never made: the
parts under test are the pure ones the defects were in.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _script(name: str):
    """A script module, importable the way `python scripts/<name>.py` makes it (its own
    directory on sys.path, for the sibling modules it imports)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- W03: generate_synthetic.py --------------------------------------------------------------

FILT = "http://w3id.org/openeduhub/vocabs/discipline/"


@pytest.fixture(scope="module")
def synthetic():
    return _script("generate_synthetic")


def _curated(synthetic, rows: list[tuple[str, str]]) -> pd.DataFrame:
    """A curated frame with the script's own column names: (labels, display names) per row."""
    title, desc, keyw = synthetic.TEXT_COLS
    label = synthetic.LABEL_COL
    return pd.DataFrame([{title: f"Titel {i}", desc: "Beschreibung", keyw: "a, b",
                          label: uris, f"{label}_DISPLAYNAME": names}
                         for i, (uris, names) in enumerate(rows)])


def test_examples_are_rows_of_that_label_exactly(synthetic):
    """W03 (audit 2026-09-30): examples were picked by substring, so the few-shot examples
    for discipline .../040 included rows of .../04003 -- another subject's text as the
    style anchor for this one."""
    import random

    curated = _curated(synthetic, [(f"{FILT}040", "Mathematik"), (f"{FILT}04003", "Statistik")])

    examples = synthetic.real_examples(curated, f"{FILT}040", random.Random(0), k=4)

    assert [e for e in examples if "Titel 1" in e] == [], examples
    assert len(examples) == 1


def test_a_name_is_never_paired_by_position(synthetic, tmp_path):
    """W03: URIs and display names were zipped by position -- the trap the README describes.
    A display name that contains the separator splits in two and shifts every later name onto
    the wrong URI; the generator then wrote rows for one subject under another's name."""
    curated = _curated(synthetic, [(
        f"{FILT}720,{FILT}380",
        "Rechts-, Wirtschafts- und Sozialwissenschaften,Mathematik",
    )])

    names = synthetic.uri_names(curated, tmp_path / "absent.json")

    assert names.get(f"{FILT}380") in (None, "Mathematik"), names


def test_names_come_from_the_vocabulary_file_when_there_is_one(synthetic, tmp_path):
    curated = _curated(synthetic, [(f"{FILT}380", "Mathe (aus der CSV)")])
    vocabulary = tmp_path / "label_names.json"
    vocabulary.write_text('{"' + FILT + '380": "Mathematik"}', encoding="utf-8")

    assert synthetic.uri_names(curated, vocabulary)[f"{FILT}380"] == "Mathematik"


def test_generated_rows_carry_the_mark_the_app_reads(synthetic):
    """W03: rows were marked `source=synthetic`, a column nothing reads. The app knows an
    LLM-written row by `generated_for` (app/provenance.py); unmarked, these rows went into
    validation and the test split and scored the model on text written like its training."""
    from app.provenance import MARK_COLUMNS, block_marks

    row = synthetic.synthetic_row({"title": "T", "description": "D", "keywords": "K"},
                                  f"{FILT}380", "Mathematik")
    frame = pd.DataFrame([row])

    assert row["generated_for"] == f"{FILT}380"
    assert "source" not in row
    assert set(MARK_COLUMNS) & set(frame.columns)
    assert block_marks(frame, mode="train").any(), "the app does not recognise the row as generated"
