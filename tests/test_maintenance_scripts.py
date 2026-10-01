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


# --- W05: eval_holdout.py --------------------------------------------------------------------


@pytest.fixture(scope="module")
def holdout():
    return _script("eval_holdout")


def test_each_model_is_judged_on_its_text_as_it_was_trained(holdout):
    """W05 (audit 2026-09-30): the holdout text was every column once, so a model trained with
    the title twice was judged on a feature distribution it never saw -- and its thresholds,
    tuned on the other one, cut in the wrong place. Built now as training built it."""
    from app.dataset_load import combine_text_columns

    frame = pd.DataFrame([{"title": "Bruch", "description": "rechnen", "extra": "x"}])
    info = {"metadata": {"text_columns": ["title", "description"],
                         "text_column_weights": {"title": 2}}}

    assert holdout.model_texts(frame, info) == combine_text_columns(
        frame, ["title", "description"], {"title": 2}).tolist()
    assert holdout.model_texts(frame, info) == ["Bruch Bruch rechnen"]


def test_a_holdout_without_a_trained_column_is_refused(holdout):
    frame = pd.DataFrame([{"title": "Bruch"}])

    with pytest.raises(SystemExit, match="description"):
        holdout.model_texts(frame, {"metadata": {"text_columns": ["title", "description"]}})


def test_nothing_to_average_is_said_rather_than_divided(holdout, monkeypatch):
    """W05: with no label shared by every model and the holdout, or no weak one among them,
    the macro averages divided by zero after the whole holdout had been classified."""
    monkeypatch.setattr(holdout, "api", lambda *a, **k: {"results": [{"predictions": []}]})

    scores = holdout.evaluate("http://unused", "key", "m", ["text"], [set()], set())

    assert scores["macro"] is None
    assert holdout.mean([]) is None
    assert holdout.mean([0.5, 1.0]) == 0.75


# --- W06: build_hochschule_dataset.py ---------------------------------------------------------

HS = "http://w3id.org/openeduhub/vocabs/hochschulfaechersystematik/"
SCHOOL = "http://w3id.org/openeduhub/vocabs/discipline/"
NAMES = {f"{HS}n1": "Informatik", f"{HS}n2": "Physik", f"{HS}n3": "Chemie"}


def test_a_built_row_names_only_the_subjects_it_kept(tmp_path):
    """W06 (audit 2026-09-30): the vocabulary filter dropped the school subjects from the label
    column and the name column was copied whole -- so it still began with the dropped
    subject's name, and when the counts happened to line up every kept subject got its
    neighbour's name: a confident wrong name, not a missing one. Run as the script runs, on a
    staging node and an export row; the output is read back the way training pairs it."""
    import csv
    import json
    import subprocess

    from app.data import split_labels
    from app.label_names import pair_names

    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps({"isPublic": True, "properties": {
        "ccm:educationalcontext": ["http://w3id.org/openeduhub/vocabs/educationalContext/hochschule"],
        "ccm:taxonid": [f"{SCHOOL}380", f"{HS}n1"],
        "ccm:taxonid_DISPLAYNAME": ["Mathematik", "Informatik"],
        "cclom:title": ["Staging"], "ccm:wwwurl": ["https://staging.example/1"],
    }}) + "\n", encoding="utf-8")
    export = tmp_path / "export.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["properties.cclom:title", "properties.ccm:taxonid",
                         "properties.ccm:taxonid_DISPLAYNAME", "properties.ccm:oeh_taxonid_university",
                         "properties.ccm:wwwurl"])
        writer.writerow(["Export", f"{SCHOOL}380, {HS}n1, {HS}n2", "Mathematik, Informatik, Physik",
                         f"{HS}n3", "https://export.example/1"])
    out = tmp_path / "out.csv"

    subprocess.run([sys.executable, str(SCRIPTS / "build_hochschule_dataset.py"), "--jsonl", str(staging),  # noqa: S603
                    "--export", str(export), "--out", str(out)],
                   check=True, capture_output=True, text=True, timeout=120)

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert len(rows) == 2
    for row in rows:
        pairs = pair_names(split_labels(row["properties.ccm:taxonid"]),
                           split_labels(row["properties.ccm:taxonid_DISPLAYNAME"]))
        wrong = [(uri, name) for uri, name in pairs if NAMES.get(uri) != name]
        assert not wrong, f"{row['properties.cclom:title']}: {wrong}"
    staged = next(r for r in rows if r["properties.cclom:title"] == "Staging")
    assert staged["properties.ccm:taxonid_DISPLAYNAME"] == "Informatik"
