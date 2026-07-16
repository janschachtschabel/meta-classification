"""Tests for the standalone tail-enrichment script (scripts/enrich_tail.py).

The script is a dataset tool, not part of the API — but its core guarantees
(no train/holdout text leakage, per-subject caps, weak-subject selection) are
exactly the kind of silent-failure logic that must be pinned by tests.
"""

import importlib.util
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "enrich_tail", Path(__file__).parent.parent / "scripts" / "enrich_tail.py")
enrich = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SPEC and enrich)

TEXT_COLS = ["title", "keywords"]
LABEL = "label"
FILT = "uri:"


def _df(rows):
    return pd.DataFrame(rows, columns=["title", "keywords", "label"]).astype(str)


def test_weak_subjects_selected_by_f1_threshold():
    metrics = {"metrics": {"per_label_f1": {"uri:a": 0.1, "uri:b": 0.9, "uri:c": 0.44}}}
    assert enrich.select_weak(metrics, max_f1=0.45) == ["uri:a", "uri:c"]


def test_holdout_split_has_no_text_leakage_and_dedupes():
    rows = [[f"titel nummer {i}", "wort", "uri:a"] for i in range(50)]
    rows.append(["titel nummer 0", "wort", "uri:a"])  # duplicate text -> dropped
    train, holdout = enrich.split_holdout(_df(rows), TEXT_COLS, LABEL, FILT,
                                          weak={"uri:a"}, seed=1)
    train_texts = set(enrich.combined_text(train, TEXT_COLS))
    hold_texts = set(enrich.combined_text(holdout, TEXT_COLS))
    assert not train_texts & hold_texts          # the one honesty rule
    assert len(train) + len(holdout) == 50       # duplicate removed
    assert len(holdout) >= 5                     # weak subjects get real holdout mass


def test_mining_respects_caps_and_known_texts():
    known_row = ["schon bekannt eins", "x", "uri:a"]
    # known_texts uses the SAME combined-text convention as the script itself
    curated_texts = set(enrich.combined_text(_df([known_row]), TEXT_COLS))
    raw = _df(
        [known_row] +                                     # known -> skipped
        [[f"neuer text {i} mit genug laenge dabei", "wort", "uri:a"] for i in range(10)] +
        [[f"anderes fach {i} mit genug laenge dabei", "wort", "uri:b"] for i in range(3)]
    )
    mined = enrich.mine_rows(raw, TEXT_COLS, LABEL, FILT, weak={"uri:a"},
                             needed={"uri:a": 4}, known_texts=set(curated_texts),
                             min_text_len=10)
    assert len(mined) == 4                        # cap respected
    texts = list(enrich.combined_text(mined, TEXT_COLS))
    assert all("neuer text" in t for t in texts)  # known + off-target rows skipped
