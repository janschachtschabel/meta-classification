"""The code that keeps /metadata's budget promise (audit 2026-09-27, T-2).

The endpoint promises every field "within the budgets from the request", and `budget.py` is
where that happens when a text is longer than its budget — the one path no test reached
(74 % covered, the lowest module in the app). These are the edges an audit probe fed it: the
behaviour was right, it just was not pinned, so a regression there would have shipped green.
Two properties hold on every edge: the result fits, and — the port's extractive guarantee —
what is left before the ellipsis is the text's own beginning.
"""

import pytest

from app.metadata.budget import clean_title, select_sentences, shorten

_EDGES = [
    ("clause", "Die Fotosynthese, ein Prozess in Pflanzenzellen, wandelt Licht in Energie um", 40),
    ("words", "Photosynthese wandelt Lichtenergie in chemische Energie um und speichert sie", 40),
    ("one unbroken word", "Donaudampfschifffahrtsgesellschaftskapitaensmuetzenfabrik", 20),
    ("break at the start", ", beginnt mit einem Komma und ist deutlich laenger als erlaubt", 20),
]


@pytest.mark.parametrize(("case", "text", "budget"), _EDGES, ids=[e[0] for e in _EDGES])
def test_a_cut_fits_the_budget_and_keeps_the_texts_own_start(case, text, budget):
    cut = shorten(text, budget)

    assert len(cut) <= budget, f"{case}: {len(cut)} characters for a budget of {budget}"
    assert cut.endswith("…")
    assert text.startswith(cut[:-1].rstrip()), f"{case}: {cut!r} is not the text's own start"


def test_a_text_that_fits_is_returned_whole():
    assert shorten("x" * 20, 20) == "x" * 20


def test_a_cut_prefers_a_clause_boundary_over_a_word():
    """Past 40 % of the budget a clause boundary wins: the cut ends a phrase, not a word run."""
    assert shorten("Die Fotosynthese, ein Prozess in Pflanzenzellen", 40) == "Die Fotosynthese…"


def test_a_quoted_title_loses_its_quotes():
    assert clean_title("„Die Fotosynthese“", 90) == "Die Fotosynthese"


def test_when_no_sentence_fits_the_best_one_is_shortened():
    long_sentence = "Ein einziger viel zu langer Satz ueber Pflanzen und Licht."
    chosen = select_sentences([long_sentence], [1.0], 20)

    assert len(chosen) <= 20 and chosen.endswith("…")


def test_chosen_sentences_keep_document_order_whatever_their_scores():
    sentences = ["Erster Satz.", "Zweiter Satz.", "Dritter Satz."]

    assert select_sentences(sentences, [0.1, 0.9, 0.5], 100) == "Erster Satz. Zweiter Satz. Dritter Satz."
