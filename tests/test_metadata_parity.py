"""The ported generators still produce what the methods were measured to produce.

`app/metadata` is a port of the default combination from
`static-metadata-generators` (`metagen/recommended.py`), whose quality numbers
(`docs/methodenvergleich.md`) are the entire reason for choosing these three methods over the
49 others that were tried. A port that drifts keeps the numbers in the plan and loses the
behaviour they describe, and nothing about the output looks wrong when it does — a slightly
different sentence split shifts a keyword, and the result still reads like a plausible
keyword list.

So the fixture is not a recording of our own output. It was generated from the source app
*before* this port was written (`docs/plans/2026-09-26-descriptive-metadata.md`), which makes
it a specification: the five sample documents in it are the source repo's own, and the
expected values are what the original pipeline produced for them.
"""

import json
from pathlib import Path

import pytest

from app.metadata import DEFAULT_SETTINGS, generate

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "metadata_parity.json").read_text(encoding="utf-8")
)
DOCUMENTS = FIXTURE["documents"]
CASES = [pytest.param(entry, id=entry["name"]) for entry in DOCUMENTS]


def test_the_fixture_describes_the_settings_this_port_defaults_to():
    """The expected values only mean anything under the budgets they were produced with."""
    expected = FIXTURE["settings"]

    assert DEFAULT_SETTINGS.title_max == expected["title_max"]
    assert DEFAULT_SETTINGS.desc_max == expected["desc_max"]
    assert DEFAULT_SETTINGS.n_keywords == expected["n_keywords"]


@pytest.mark.parametrize("entry", CASES)
def test_the_title_matches_the_source_pipeline(entry):
    """`title.heading_or_keywords`: a real heading at the top of the text, else the template
    from the three best keywords. Both branches are represented in the fixture — four
    documents open with a heading, and the keyword template is exercised by
    `test_a_text_without_a_heading_falls_back_to_the_keyword_template` below."""
    assert generate(entry["text"]).title == entry["title"]


@pytest.mark.parametrize("entry", CASES)
def test_the_description_matches_the_source_pipeline(entry):
    """`description.lead`: the first usable sentences up to the character budget.

    This is the one place the port deviates from the source app's own default, which is
    `description.lead_centroid` — that one downloads a 1 GB Model2Vec model at runtime, which
    `test_no_url_fetch.py` exists to prevent, and it scored *lower* on the real-material test
    set (1,0 against 1,2). The reasoning is in the plan; what is asserted here is that the
    method we did port is ported faithfully.
    """
    assert generate(entry["text"]).description == entry["description"]


@pytest.mark.parametrize("entry", CASES)
def test_the_keywords_match_the_source_pipeline(entry):
    """`keywords.tfidf`: noun phrases by frequency in the text and rarity in German."""
    assert generate(entry["text"]).keywords == entry["keywords"]


@pytest.mark.parametrize("entry", CASES)
def test_the_description_and_title_respect_their_budgets(entry):
    """The budgets are the contract a consumer writes its database column against."""
    result = generate(entry["text"])

    assert len(result.title) <= DEFAULT_SETTINGS.title_max
    assert len(result.description) <= DEFAULT_SETTINGS.desc_max
    assert len(result.keywords) <= DEFAULT_SETTINGS.n_keywords


def test_the_one_documented_stemmer_difference_is_still_only_an_ordering():
    """api_v3 ships `snowballstemmer`, the source app uses `nltk`; the two implement
    different Snowball variants for German and their stems differ on 2,66 % of a 627k-word
    vocabulary.

    nltk was rejected because importing it costs 10,4 s and pulls `urllib.request`, `socket`
    and `ssl` into a process that is architecturally forbidden from fetching URLs. The whole
    measured price of that decision is one document ranking the same eight keywords in a
    different order, and this pins it: if a future change makes the two disagree on WHICH
    keywords come out, that is a different and much larger claim than the plan makes.
    """
    drifted = [entry for entry in DOCUMENTS if "keywords_nltk_order" in entry]

    assert len(drifted) == 1, (
        f"the plan documents exactly one document whose keyword ORDER differs between the two "
        f"stemmers; the fixture now has {len(drifted)}: {[e['name'] for e in drifted]}"
    )
    for entry in drifted:
        produced = generate(entry["text"]).keywords
        assert set(produced) == set(entry["keywords_nltk_order"]), (
            f"{entry['name']}: the stemmers now disagree on WHICH keywords are produced, not "
            f"only on their order"
        )


def test_a_text_without_a_heading_falls_back_to_the_keyword_template():
    """The second branch of `title.heading_or_keywords`, which is the one that carries running
    text: with no heading to take, the title is built as "K1: K2 und K3" from the three best
    keywords. This is why the method was chosen over plain heading extraction, which drops to
    the level of the first sentence on prose (sem 0,82 -> 0,58)."""
    # The fixture's documents all open with a heading, so the fallback needs prose: one
    # paragraph, no line that could pass as a title.
    prose = (
        "Die Fotosynthese ist der Vorgang, mit dem Pflanzen aus Licht, Wasser und "
        "Kohlenstoffdioxid Traubenzucker aufbauen. Dabei entsteht Sauerstoff, den Menschen "
        "und Tiere zum Atmen brauchen. Das Chlorophyll in den Blättern nimmt das Licht auf "
        "und treibt die Reaktion an. Ohne Fotosynthese gäbe es auf der Erde keine "
        "Nahrungsketten, denn alle Tiere leben davon, dass Pflanzen Energie speichern."
    )
    title = generate(prose).title

    assert ":" in title and " und " in title, f"not the keyword template: {title!r}"
    # It names the subject rather than a passing detail, and uses only words from the text.
    assert "Fotosynthese" in title
