"""Markup removal for the descriptive-metadata generators.

`/predict` flattens HTML and Markdown to one line of space-separated tokens, because that is
what the vectorizer was fitted on. The generators here need the same markup gone but the line
breaks kept, because a heading is a heading by virtue of standing alone on its line. The
patterns are therefore shared (`app.markup`) and the two pipelines compose them differently.

What is pinned here is that composition: markup does not reach the output, a block-level tag
ends a line, and the aggressive half of the Markdown cleaning — the one that would turn
`Name: ______ Datum: ______` into `Name: Datum:` — stays out of this pipeline.
"""

import time

import pytest

from app import data
from app.metadata import generate
from app.metadata.textprep import clean_text


def test_an_html_heading_becomes_the_title_without_its_tag():
    html = ("<h1>Bruchrechnung im Alltag</h1>\n"
            "<p>Brüche begegnen uns im Alltag häufig: beim Teilen einer Pizza oder in "
            "Rezepten. Wer sie sicher addieren kann, rechnet im Alltag schneller.</p>")

    assert generate(html).title == "Bruchrechnung im Alltag"


def test_a_block_tag_ends_a_line_even_without_a_newline():
    """The whole reason this pipeline needs its own composition.

    Scraped HTML frequently has no newline between the heading and the body. If a tag became
    a space, the two would merge into one line and the heading detection would return the
    heading plus the first sentence. A block-level tag has to become a line break.
    """
    one_line = ("<h1>Bruchrechnung im Alltag</h1><p>Brüche begegnen uns im Alltag häufig: "
                "beim Teilen einer Pizza oder in Rezepten.</p>")

    assert generate(one_line).title == "Bruchrechnung im Alltag"


def test_entities_are_decoded():
    assert "&" in clean_text("Zähler &amp; Nenner")
    assert "&amp;" not in clean_text("Zähler &amp; Nenner")


def test_a_markdown_heading_marker_is_removed():
    assert generate("# Bruchrechnung\n\nBrüche begegnen uns im Alltag häufig.").title == (
        "Bruchrechnung"
    )


def test_markdown_emphasis_is_removed():
    assert clean_text("Der **Hauptnenner** ist *wichtig*.") == "Der Hauptnenner ist wichtig."


def test_a_markdown_link_keeps_its_text():
    assert clean_text("Siehe [die Anleitung](http://example.org/x).") == (
        "Siehe die Anleitung."
    )


def test_a_form_blank_is_not_mistaken_for_emphasis():
    """`app.data.clean_text` may strip a long run of underscores; this pipeline may not.

    Measured on `tests/fixtures/metadata_parity.json` document 02, a real worksheet: the line
    `Name: ____________________   Datum: ____________` is 21% letters, so the letter-ratio
    filter drops it. Strip the underscores first and it becomes `Name: Datum:` — 82% letters,
    kept, and it lands in the description as a line that says nothing. A run of four or more
    identical markers is a form blank or a rule, never emphasis.
    """
    blank = "Name: ____________________   Datum: ____________"

    assert clean_text(f"Arbeitsblatt Bruchrechnung\n{blank}\nBrüche kommen im Alltag vor.") == (
        "Arbeitsblatt Bruchrechnung\nBrüche kommen im Alltag vor."
    )


def test_a_run_of_four_markers_is_left_alone():
    """The boundary the case above turns on: three or fewer is emphasis, four or more is not.

    A line that is nothing but a rule never reaches this question — it has no letters and the
    letter-ratio filter drops it. What matters is a run sitting in a line of real text, which
    survives that filter and would otherwise be silently edited.
    """
    assert "____" in clean_text("Der Hauptnenner ____ ist gesucht und wird berechnet.")


def test_an_inline_comparison_survives():
    """`>` and `#` are only markup at the start of a line."""
    assert clean_text("Für a > b gilt das nicht. Siehe Aufgabe #5.") == (
        "Für a > b gilt das nicht. Siehe Aufgabe #5."
    )


def test_a_blockquote_marker_is_removed_at_the_start_of_a_line():
    assert clean_text("> Brüche begegnen uns im Alltag häufig.") == (
        "Brüche begegnen uns im Alltag häufig."
    )


def test_markup_removal_does_not_backtrack():
    """Same budget and the same reason as `test_clean_text_does_not_backtrack_on_unclosed_markup`.

    The block-tag pattern excludes `<` from its body for exactly that reason: an unclosed run
    must let a start position fail in constant time instead of scanning to the end of the
    string once per position. `<p` repeated is the case the new pattern adds — a run of
    half-written block tags.

    Looped rather than parametrized because a 50,000-character test id does not fit in the
    environment variable pytest passes it through.
    """
    hostile = [
        "<" * 50_000,
        "<p" * 50_000,
        "[" * 50_000,
        # The three Markdown patterns this pipeline adds. A heading line whose body is one long
        # whitespace run and whose last character is neither space nor `#` made the lazy body,
        # the optional closing run and the trailing whitespace all compete for the same
        # characters: 9.2 s at 25,000 and quadratic from there, on a line a caller may send.
        "# a" + " " * 50_000 + "x",
        "> a" + " " * 50_000 + "x",
        "*a" + " " * 50_000 + "x",
        "#" * 50_000,
        ">" * 50_000,
        "*" * 50_000,
        "_" * 50_000,
    ]
    for case in hostile:
        started = time.perf_counter()
        clean_text(case)
        assert time.perf_counter() - started < 2.0, f"slow on {case[:3]!r} run"


def test_the_vectorizer_pipeline_still_flattens():
    """The property `app.data.clean_text` must not lose to the shared patterns.

    Its output feeds a fitted vectorizer, so it has to stay one line of single-spaced tokens
    even though the patterns it now shares turn a block tag into a newline.
    """
    flattened = data.clean_text("<h1>Titel</h1><p>Erster Satz.</p><p>Zweiter Satz.</p>")

    assert flattened == "Titel Erster Satz. Zweiter Satz."


def test_a_script_body_does_not_reach_the_output():
    """Embedded code is not markup, and removing the tags around it is not enough.

    Measured while adding this step: with the tags gone but the body kept, `<h1>Titel</h1>
    <script>var t = {id: 42};</script>` put the JavaScript into the title, because the script
    body landed on the heading's line. A script or style body is never prose — it goes.
    """
    page = ("<h1>Bruchrechnung</h1><script>var tracker = {id: 42}; function send(){ "
            "return fetch('/x'); }</script>"
            "<p>Brüche begegnen uns im Alltag häufig beim Teilen einer Pizza.</p>")

    result = generate(page)

    assert result.title == "Bruchrechnung"
    assert "tracker" not in result.description
    assert "function" not in result.description


def test_a_style_body_does_not_reach_the_output():
    page = ("<h1>Bruchrechnung</h1><style>.brueche { color: #ff0000; font-weight: bold; }"
            "</style><p>Brüche begegnen uns im Alltag häufig.</p>")

    result = generate(page)

    assert result.title == "Bruchrechnung"
    assert "color" not in result.description


def test_an_unclosed_script_tag_loses_only_its_tag():
    """This test used to pin the opposite: an unclosed `<script>` took the rest of the text
    with it, so that a page truncated mid-script left no code behind. But a text ABOUT HTML
    says `<script>` in its prose, and lost everything after the word -- description and
    keywords came from what was left (audit 2026-09-30, M05). A body goes only with its
    closing tag now; a truncated page keeps a line of code, which the sentence filters see.
    Not hanging is pinned by `test_script_removal_does_not_backtrack`."""
    cleaned = clean_text("<h1>Titel</h1><script>var x = 1;")

    assert cleaned.startswith("Titel")
    assert "<script>" not in cleaned


def test_a_closed_atx_heading_loses_both_marker_runs():
    assert clean_text("### Bruchrechnung im Alltag ###") == "Bruchrechnung im Alltag"


def test_a_hash_that_is_not_a_heading_marker_survives():
    """Closed-ATX handling only applies to a line that opens with `#`."""
    assert clean_text("Die Antwort steht in Aufgabe #5 #") == "Die Antwort steht in Aufgabe #5 #"


def test_a_prose_mention_of_a_tag_is_not_treated_as_one():
    """Why script removal runs before entities are decoded.

    Decode first and `Ein &lt;script&gt;-Tag ...` reads as a real unclosed `<script>`, whose
    body is the rest of the sentence — the removal would delete the text it is describing.
    """
    cleaned = clean_text("Ein &lt;script&gt;-Tag im Quelltext ist ein Sicherheitsrisiko.")

    # The sentence as its author wrote it: `&lt;script&gt;` MEANS the characters `<script>`.
    # (It used to assert `<script>` was absent -- an artifact of decoding before the tags went.)
    assert cleaned == "Ein <script>-Tag im Quelltext ist ein Sicherheitsrisiko."


def test_script_removal_does_not_backtrack():
    """The pattern's whole reason for being written as an unrolled loop.

    A lazy `.*?` body rescans to the end of the document once per unclosed `<script`, which is
    quadratic; 12,500 of them in a 100,000-character text is the maximum a request may carry.
    """
    for hostile in ("<script>" * 12_500, "<script>x" * 10_000, "<style>" * 14_000):
        started = time.perf_counter()
        clean_text(hostile)
        assert time.perf_counter() - started < 2.0, f"slow on {hostile[:8]!r} run"


def test_a_quoted_heading_loses_both_markers():
    """The quote marker has to go first: `_MD_HEADING_RE` is anchored to `^`, so a line still
    starting with `>` is not a heading to it and keeps its `#`."""
    assert clean_text("> # Bruchrechnung im Alltag") == "Bruchrechnung im Alltag"


def test_nested_quote_markers_all_go():
    assert clean_text(">> Brüche begegnen uns im Alltag häufig.") == (
        "Brüche begegnen uns im Alltag häufig."
    )


@pytest.mark.parametrize("marked, expected", [
    ("**Hauptnenner**bestimmung", "Hauptnennerbestimmung"),
    ("**fett**_kursiv_", "fettkursiv"),
    ("~~weg~~**fett**", "wegfett"),
    ("**A**1", "A1"),
    ("**fett**`code`", "fettcode"),
])
def test_emphasis_markers_go_when_a_word_follows_them_directly(marked, expected):
    """The hole the space-delimited cases could not reach.

    `**foo**bar` is valid CommonMark — adjacent inline spans need no separator — and German
    compounds land in exactly that shape ("**Bruch**rechnung"). The intra-word keep below was
    length-agnostic, so a `**` between two word characters survived into the proposed title:
    `**Hauptnenner**bestimmung` came back as `Hauptnenner**bestimmung`.

    The parametrized test above cannot catch this: its frame puts whitespace on both sides of
    the run, so it never reaches the intra-word branch at all.
    """
    assert clean_text(f"Der {marked} ist gesucht.") == f"Der {expected} ist gesucht."


def test_a_double_underscore_inside_a_word_survives():
    """Why the fix is not simply "keep only single-character runs".

    That would strip `__` from an identifier. CommonMark's own rule is about the character,
    not the length: `_` is not emphasis between word characters, while `*` is — so an
    all-underscore run is kept whatever its length.
    """
    assert clean_text("Die Variable my__var ist gesetzt.") == "Die Variable my__var ist gesetzt."


def test_an_underscore_inside_a_word_survives():
    """The endpoint guarantees every word it returns occurs in the input.

    `arbeitsblatt_1_loesung.pdf` becoming `arbeitsblatt1loesung.pdf` breaks that guarantee —
    the returned word is one the text never contained. CommonMark agrees: `_` between word
    characters is not emphasis.
    """
    assert clean_text("Die Datei arbeitsblatt_1_loesung.pdf gehört dazu.") == (
        "Die Datei arbeitsblatt_1_loesung.pdf gehört dazu."
    )
    assert clean_text("Siehe https://example.org/a_b_c für mehr.") == (
        "Siehe https://example.org/a_b_c für mehr."
    )


@pytest.mark.parametrize("marked", [
    "**_Hauptnenner_**",
    "~~**Hauptnenner**~~",
    "***Hauptnenner***",
    "**`Hauptnenner`**",
    "~~*Hauptnenner*~~",
    "~~Hauptnenner~~",
    "*Hauptnenner*",
    "`Hauptnenner`",
])
def test_emphasis_markers_go_whether_the_run_is_mixed_or_not(marked):
    """A run of markers is emphasis whatever it is made of.

    The rule is deliberately about the run, not about matching pairs: a lookaround version
    required the markers on both sides to be the SAME character, which left every combined
    form (`**_fett_**`, `~~**weg**~~`) untouched and its markers in the proposed title. Only
    two of these shapes were pinned before, so the other six were working by luck rather than
    by test.
    """
    assert clean_text(f"Der {marked} ist gesucht.") == "Der Hauptnenner ist gesucht."


def test_navigation_and_footer_content_is_not_prose():
    """Found by an end-to-end check, not by reasoning.

    A scraped page opens with a breadcrumb: `<nav><a>Startseite</a> » <a>Mathematik</a></nav>`.
    Remove only the tags and that becomes the document's first line — short, capitalised, no
    final period — which is exactly what the title heuristic looks for, so the proposed title
    came back as `Startseite » Mathematik` instead of the `<h1>` right below it.

    `<nav>` and `<footer>` are the two elements HTML defines as chrome rather than content, so
    their bodies go the way a script body does. `<header>` and `<aside>` deliberately stay:
    pages put the real `<h1>` inside `<header>`, and an `<aside>` often holds a definition box.
    """
    page = ('<nav><a href="/">Startseite</a> &raquo; <a href="/mathe">Mathematik</a></nav>'
            "<h1>Brüche addieren</h1>"
            "<p>Brüche begegnen uns im Alltag häufig: beim Teilen einer Pizza.</p>"
            "<footer>Kontakt: redaktion@example.org</footer>")

    result = generate(page)

    assert result.title == "Brüche addieren"
    assert "Startseite" not in result.description
    assert "redaktion@example.org" not in result.description


def test_a_heading_inside_a_header_element_still_counts():
    """The reason `<header>` is not treated as chrome."""
    page = ("<header><h1>Brüche addieren</h1></header>"
            "<p>Brüche begegnen uns im Alltag häufig: beim Teilen einer Pizza.</p>")

    assert generate(page).title == "Brüche addieren"



# --- M05 (audit 2026-09-30): markup handling that invented sentences and cut words -----------


def test_an_escaped_comparison_is_prose_not_a_tag():
    """Entities were decoded BEFORE the tags went, so `&lt; ... &gt;` became a tag that took
    the prose between with it -- and the description was a sentence the text does not have."""
    text = ("Für alle x &lt; 5 gilt die erste Regel, und für y &gt; 3 gilt die zweite Regel. "
            "Beide Regeln stehen im Lehrbuch auf Seite zwölf.")

    assert clean_text(text).startswith("Für alle x < 5 gilt die erste Regel, und für y > 3")
    assert "Für alle x 3 gilt" not in generate(text).description


def test_a_spaced_comparison_is_prose_not_a_tag():
    """A tag starts with `<` and a letter, `/`, `!` or `?` -- never with a space or a digit."""
    assert clean_text("Wenn a < b und c > d, dann gilt a + c < b + d.") == (
        "Wenn a < b und c > d, dann gilt a + c < b + d.")


def test_a_literal_script_tag_in_prose_keeps_the_rest_of_the_text():
    text = ("Das <script>-Element lädt JavaScript in eine Seite.\n"
            "Der zweite Absatz erklärt, warum Formulare validiert werden müssen.")

    assert "Der zweite Absatz erklärt" in clean_text(text)


def test_a_commented_out_banner_is_not_text():
    text = ("<!-- <h1>Werbung: Jetzt kaufen</h1> -->\n<h1>Bruchrechnung</h1>\n"
            "<p>Brüche werden erweitert, gekürzt und addiert.</p>")

    assert "Werbung" not in clean_text(text)
    assert generate(text).title == "Bruchrechnung"


def test_an_inline_tag_does_not_split_a_word():
    """A browser renders `<b>Bruch</b>rechnung` as one word, and `H<sub>2</sub>O` as H2O."""
    assert clean_text("<p><b>Bruch</b>rechnung mit H<sub>2</sub>O</p>") == "Bruchrechnung mit H2O"


def test_a_closed_heading_with_windows_line_endings_loses_its_markers():
    """`\\r\\n` left the `\\r` inside the heading line, so the closing run survived as the
    title: `Bruchrechnung ###`. Line endings are normalised before any markup rule runs."""
    assert clean_text("### Bruchrechnung ###\r\nBrüche werden gekürzt.").split("\n")[0] == "Bruchrechnung"


def test_an_escaped_markdown_marker_stays_a_character():
    """Entities are decoded last, so an escaped `#` is the author's character, not a heading."""
    assert clean_text("&#35; ist das Zeichen für eine Nummer.") == "# ist das Zeichen für eine Nummer."


def test_comment_and_element_removal_take_linear_time():
    """The two new passes find their closing delimiters in Python: an unclosed opener must
    not rescan the rest of the text once per occurrence."""
    for hostile in ("<!--" * 25_000, "<!-- x" * 16_000, "<nav>" * 20_000, "<footer>x" * 10_000,
                    "<script>" * 6_000 + "</script>"):
        started = time.perf_counter()
        clean_text(hostile)
        assert time.perf_counter() - started < 2.0, f"slow on {hostile[:8]!r} run"
