"""HTML and Markdown removal for the two text pipelines.

Both pipelines have to get markup out of the way, and they need different things afterwards:

- `app.data.clean_text` flattens everything to one line of space-separated tokens, because
  that is exactly what a fitted vectorizer saw. It composes the patterns itself and has to
  keep composing them in its original order — see `strip_tags` and `CONTROL_RE` for the two
  places where a different order silently changes its output.
- `strip_markup_preserving_lines` keeps the line breaks: a heading is a heading by virtue of
  standing alone on its line, and the descriptive-metadata generators recover headings,
  wrapped lines and page chrome from that structure.

Every pattern here excludes its own opening delimiter from the body it scans, and the two
that would otherwise be ambiguous are resolved in Python rather than by backtracking. That is
load-bearing, not cosmetic: this project has twice paid for a quadratic pattern reachable from
a request — `<[^>]+>` cost 52 s for one 100,000-character text, and an ambiguous heading
pattern cost 9.2 s at 25,000 characters. Both hold the GIL throughout, so running in a worker
thread is no protection. A new pattern here needs a scaling measurement;
`tests/test_metadata_markup.py` pins the budget.
"""

from __future__ import annotations

import html
import re

# Public because `app.data.clean_text` applies it LAST, after the Markdown passes, and the
# order matters. `\x92` is what a cp1252-mis-decoded curly quote leaves behind and is
# plausible in scraped rows; removed first, `[Titel]\x92(http://x)` becomes a well-formed
# Markdown link that then collapses to `Titel`, where removed last it keeps both brackets.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Block-level elements end a line; everything else is inline and becomes a space. Scraped
# HTML routinely has no newline between a heading and the body that follows it, so without
# this distinction the two merge into one line and a heading stops being recognisable as one.
_BLOCK_TAG_NAMES = (
    "address|article|aside|blockquote|body|br|caption|dd|details|div|dl|dt|fieldset"
    "|figcaption|figure|footer|form|h[1-6]|head|header|hr|legend|li|main|nav|noscript|ol"
    "|option|p|pre|section|summary|table|tbody|td|tfoot|th|thead|tr|ul"
)
# The lookahead keeps `<pre>` from matching inside `<prefix>`: after the name there has to be
# whitespace, a `/` or the closing `>`.
BLOCK_TAG_RE = re.compile(rf"</?(?:{_BLOCK_TAG_NAMES})(?=[\s/>])[^<>]*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^<>]+>")
# Four elements whose body is not prose, so removing the tags around them is not enough:
# `script` and `style` hold code, and `nav` and `footer` are what HTML calls chrome rather than
# content. Measured end to end: with only the tags removed, a page opening
# `<nav><a>Startseite</a> » <a>Mathematik</a></nav>` made that breadcrumb the document's first
# line, and the title heuristic — which looks for a short capitalised line without a final
# period — proposed `Startseite » Mathematik` over the `<h1>` right below it.
#
# `header` and `aside` are deliberately absent: pages put the real `<h1>` inside `<header>`, and
# an `<aside>` often carries a definition box. Only `strip_markup_preserving_lines` applies any
# of this — `clean_text` feeds fitted vectorizers, and dropping a body it previously kept would
# shift the features of every model already trained on scraped HTML.
#
# Written as an unrolled loop rather than `.*?` so it cannot backtrack: the body alternates
# runs of non-`<` characters with a `<` that is not the closing tag, every iteration consumes
# at least that one character, and `[^<]*` always stops at the next `<`. There is exactly one
# way to parse any prefix. A lazy `.*?` would instead rescan to the end of the document once
# per unclosed `<script`. The closing tag is optional so a page truncated mid-element loses the
# remainder instead of keeping it.
NON_PROSE_ELEMENT_RE = re.compile(
    r"<(script|style|nav|footer)\b[^<>]*>"
    r"[^<]*(?:<(?!/\1\b)[^<]*)*"
    r"(?:</\1\b[^<>]*>)?",
    re.IGNORECASE,
)
MD_LINK_RE = re.compile(r"!?\[([^\[\]]*)\]\([^()]*\)")  # [text](url) / ![alt](url) -> text/alt
MD_MARK_RE = re.compile(r"[*_`~#>]+")  # emphasis / code / heading / quote markers

# `>` is markup only where a line starts; inline it is a comparison ("für a > b"). Repeated so
# a nested quote clears in one pass.
_MD_QUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[ \t]?)+", re.MULTILINE)
# Likewise `#`. The body is captured greedily and trimmed in `_heading` rather than with an
# optional trailing group in the pattern — see there.
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*)$", re.MULTILINE)
_MD_MARKER_RUN_RE = re.compile(r"[*_`~]+")


def _heading(match: re.Match[str]) -> str:
    r"""The text of an ATX heading, without a closed-ATX trailing run (`### Titel ###`).

    Trimmed here rather than in the pattern. Written as
    `#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$` the tail is ambiguous — the lazy body, the
    optional closing run and the trailing whitespace all compete for the same characters — so
    on a line ending in anything that is neither space nor `#`, every expansion of the body
    re-scans the whitespace run twice before failing. Measured 9.2 s for one 25,000-character
    line and quadratic beyond, on a request that may carry 100,000. Two rstrips are one
    linear scan.
    """
    return match.group(1).rstrip(" \t").rstrip("#").rstrip(" \t")


def _emphasis(match: re.Match[str]) -> str:
    """Drop a run of Markdown emphasis markers, unless it is not emphasis."""
    run = match.group()
    # A long run of ONE repeated marker is a rule or a form blank. Measured on parity fixture
    # 02, a real worksheet: `Name: ____________________   Datum: ____________` is 21% letters
    # so `letter_ratio` drops it, but strip the underscores and it becomes `Name: Datum:` —
    # 82% letters, kept, and a line that says nothing lands in the description.
    if len(run) > 3 and len(set(run)) == 1:
        return run
    # Between two word characters a marker can still belong to the word, but only in the two
    # shapes that are not emphasis. `/metadata` guarantees every word it returns occurs in the
    # input, so rewriting `arbeitsblatt_1_loesung.pdf` would break the contract outright.
    #   - An all-underscore run, whatever its length: CommonMark does not treat `_` as emphasis
    #     between word characters, which is the rule that keeps `a_b_c` and `my__var` intact.
    #   - A single character of any marker: `2*3` is multiplication, and one delimiter has
    #     nothing to pair with. `*a*b*c*` therefore keeps its middle `*`, which is inherent —
    #     telling it from `2*3` needs a real parser, not a pattern.
    # Anything longer IS emphasis: `**foo**bar` is valid CommonMark (adjacent spans need no
    # separator) and German compounds land in that shape, so a length-agnostic keep put the
    # markers straight into a proposed title — `**Hauptnenner**bestimmung` came back as
    # `Hauptnenner**bestimmung`.
    start, end = match.span()
    before = match.string[start - 1] if start else ""
    after = match.string[end] if end < len(match.string) else ""
    if before.isalnum() and after.isalnum() and (set(run) == {"_"} or len(run) == 1):
        return run
    return ""


def _tag(match: re.Match[str]) -> str:
    return "\n" if BLOCK_TAG_RE.fullmatch(match.group()) else " "


def strip_tags(text: str) -> str:
    r"""Decode entities and remove HTML tags; a block-level tag becomes a line break.

    One pass that decides the replacement per match, rather than a block-tag pass followed by
    a general one. Substituting block tags separately deletes the `<` and `>` that stop
    `HTML_TAG_RE`'s `[^<>]+`, and the second pass then joins a bare `<` before the tag to a
    bare `>` after it and swallows the prose between them: `Preis < 5 Euro <br> Menge > 3
    Stück` came back as `Preis 3 Stück`. Matching exactly the set that a single tag pass
    always matched is what lets `app.data.clean_text` share this without changing what it
    produces — whether a tag became `"\n"` or `" "` is invisible once its whitespace collapse
    runs.
    """
    return HTML_TAG_RE.sub(_tag, html.unescape(text))


def strip_markup_preserving_lines(raw: str) -> str:
    """Remove HTML and Markdown from prose, leaving the line structure intact.

    What `app.metadata.textprep.clean_text` composes onto. `app.data.clean_text` deliberately
    does NOT use this: it applies `MD_MARK_RE`, which flattens marker runs of any length and
    anywhere inside a word, and it keeps script and style bodies. Both differences would
    change what an already-fitted vectorizer sees, and neither matters once its whitespace
    collapse has run.
    """
    # Non-prose bodies go before `strip_tags` decodes entities: afterwards a prose mention of
    # "&lt;script&gt;" would read as a real unclosed tag and take the rest of the sentence with
    # it. A line break in their place keeps a heading off the next block.
    text = NON_PROSE_ELEMENT_RE.sub("\n", raw)
    text = strip_tags(text)
    text = CONTROL_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    # Quote markers before headings: `_MD_HEADING_RE` is anchored to `^`, so a line that still
    # starts with `>` is not a heading to it and would keep its `#`.
    text = _MD_QUOTE_RE.sub("", text)
    text = _MD_HEADING_RE.sub(_heading, text)
    return _MD_MARKER_RUN_RE.sub(_emphasis, text)
