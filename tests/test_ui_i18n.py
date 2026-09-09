"""The UI speaks German and English, and neither language may rot.

The admin UI has no build step and no test runner of its own (vanilla JS on
purpose), so the guards that a JS toolchain would give us live here instead:
the string maps are plain JSON inside a one-line assignment, which lets this
suite read them the way a translator does — as data.

Four things can go wrong with a hand-maintained i18n layer, and each has a test
below: a key exists in one language only, the markup or the code asks for a key
nobody defined, the map grows entries nothing uses, and — the one that made D6
worth doing — a string never made it into the map at all.
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

UI = Path(__file__).parent.parent / "app" / "static" / "ui"
INDEX = UI / "index.html"

# Every element attribute the applier understands. Kept here as the single list so
# a new one cannot be added to i18n.js without this suite learning about it.
I18N_ATTRIBUTES = ("data-i18n", "data-i18n-html", "data-i18n-placeholder",
                   "data-i18n-title", "data-i18n-aria-label")
# Attributes whose value a person reads. Each needs its own data-i18n-* twin.
TRANSLATABLE_ATTRIBUTES = ("placeholder", "title", "aria-label", "alt")

_ASSIGNMENT_RE = re.compile(r"^\s*const\s+STRINGS_[A-Z]{2}\s*=\s*", re.MULTILINE)
_KEY_IN_HTML_RE = re.compile(r"data-i18n(?:-[a-z-]+)?=\"([^\"]+)\"")
# Not every key reaches t() as a literal: a table of columns or form fields carries
# its keys as data. Matching the namespace instead of the call site finds those too —
# and is why no key in this UI may be assembled from a template.
NAMESPACES = ("common", "errors", "shell", "login", "query", "explain", "feedback",
              "share", "train", "trainStatus", "models", "modelDetail", "evaluate",
              "datasets", "datasetDetail")
_KEY_IN_JS_RE = re.compile(
    r"[\"'](" + "|".join(NAMESPACES) + r")((?:\.[A-Za-z0-9]+)+)[\"']")


def load_map(language: str) -> dict:
    """The string map of one language, parsed as the JSON it has to be.

    The map file is data with a two-line JS wrapper so the browser can load it
    with a <script> tag (no fetch, no async boot, no half-translated first
    paint). Parsing it as strict JSON here is what keeps it data: a comment, a
    trailing comma or a computed value would fail this and be caught at once.
    """
    text = (UI / f"strings-{language}.js").read_text(encoding="utf-8")
    body = _ASSIGNMENT_RE.split(text, maxsplit=1)[-1].strip()
    assert body.endswith(";"), f"strings-{language}.js must end its assignment with ';'"
    return json.loads(body[:-1])


def ui_scripts() -> list[Path]:
    """The UI's own JS — not the string maps, not the vendored Swagger bundle."""
    return sorted(p for p in UI.glob("*.js") if not p.name.startswith("strings-"))


def strip_js_comments(source: str) -> str:
    """Remove /* */ and // comments so a comment cannot satisfy or trip a check."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", source)


def keys_the_ui_uses() -> set[str]:
    used = set(_KEY_IN_HTML_RE.findall(INDEX.read_text(encoding="utf-8")))
    for script in ui_scripts():
        source = strip_js_comments(script.read_text(encoding="utf-8"))
        used |= {head + tail for head, tail in _KEY_IN_JS_RE.findall(source)}
    return used


def test_both_languages_define_exactly_the_same_keys():
    de, en = load_map("de"), load_map("en")
    assert set(de) == set(en), (
        "the two string maps drifted apart — "
        f"only German: {sorted(set(de) - set(en))}; only English: {sorted(set(en) - set(de))}"
    )


def test_every_key_the_ui_asks_for_is_defined():
    defined = set(load_map("de"))
    used = keys_the_ui_uses()
    assert used, "no i18n keys found at all — the extraction regexes have gone stale"
    assert not used - defined, f"used but never defined: {sorted(used - defined)}"


def test_no_string_is_defined_but_unused():
    """An orphan key is a translation nobody reads and nobody notices going wrong."""
    defined = set(load_map("de"))
    assert not defined - keys_the_ui_uses(), (
        f"defined but never used: {sorted(defined - keys_the_ui_uses())}"
    )


def test_plural_entries_carry_every_form_both_languages_need():
    """A count-bearing entry is an object of plural categories, not a string.

    German and English both distinguish only one/other, so those two are the
    contract. Anything else in there is a category ``Intl.PluralRules`` will
    never ask for on this UI.
    """
    for language in ("de", "en"):
        for key, value in load_map(language).items():
            if isinstance(value, dict):
                assert set(value) == {"one", "other"}, (
                    f"{language}:{key} is a plural entry with {sorted(value)}; "
                    "it must carry exactly 'one' and 'other'"
                )


def test_markup_entries_take_no_parameters():
    """``data-i18n-html`` values reach innerHTML, so they must be constants.

    The applier passes no parameters for them on purpose — a placeholder here
    would be the seam through which caller data could reach innerHTML unescaped.
    Assert the strings themselves cannot want one.
    """
    html_keys = set(re.findall(r"data-i18n-html=\"([^\"]+)\"", INDEX.read_text(encoding="utf-8")))
    for language in ("de", "en"):
        strings = load_map(language)
        for key in html_keys:
            assert "{" not in strings[key], (
                f"{language}:{key} is rendered as markup but carries a placeholder"
            )


class _ProseScanner(HTMLParser):
    """Find text and human-readable attributes in the markup that no key covers.

    Text is "covered" when the element it sits in, or one of its ancestors,
    carries a ``data-i18n`` hook — an ancestor counts because a whole prose
    block (a help ``<details>``, a list) is translated as one string.
    """

    NO_CLOSING_TAG = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                      "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.covered: list[bool] = [False]
        self.skipping = 0            # inside <script>/<style>: not prose
        self.loose_text: list[str] = []
        self.loose_attributes: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name in TRANSLATABLE_ATTRIBUTES:
            if attributes.get(name, "").strip() and f"data-i18n-{name}" not in attributes:
                self.loose_attributes.append(f"<{tag} {name}={attributes[name]!r}>")
        hooked = any(name in attributes for name in ("data-i18n", "data-i18n-html"))
        if tag in self.NO_CLOSING_TAG:
            return
        if tag in ("script", "style"):
            self.skipping += 1
        self.covered.append(self.covered[-1] or hooked)

    def handle_endtag(self, tag):
        if tag in self.NO_CLOSING_TAG:
            return
        if tag in ("script", "style") and self.skipping:
            self.skipping -= 1
        if len(self.covered) > 1:
            self.covered.pop()

    def handle_data(self, data):
        if self.skipping or self.covered[-1] or not data.strip():
            return
        self.loose_text.append(" ".join(data.split()))


def test_no_prose_is_left_hardcoded_in_the_markup():
    """The D6 gate: every word a person reads comes out of the string map.

    An untranslated string does not fail loudly — it just quietly stays English
    in a German UI, which is exactly the state the owner asked us to leave
    behind. This test is what makes "all of it" checkable rather than claimed.
    """
    scanner = _ProseScanner()
    scanner.feed(INDEX.read_text(encoding="utf-8"))
    assert not scanner.loose_text, f"untranslated text in index.html: {scanner.loose_text}"
    assert not scanner.loose_attributes, (
        f"untranslated attributes in index.html: {scanner.loose_attributes}"
    )


# A sentence-shaped literal is one that opens with a capitalised word followed by
# a lowercase one — how every user-facing string in this UI is written, and almost
# nothing else. A net rather than a proof: markup fragments and lowercase labels
# slip through it, so the diff still has to be read.
_SENTENCE_RE = re.compile(r"[\"'`]([A-ZÄÖÜ][a-zäöüß]+ [a-zäöüß]{2,}[^\"'`]*)")


def test_no_sentence_shaped_literal_is_left_in_the_ui_code():
    offenders = {}
    for script in ui_scripts():
        found = _SENTENCE_RE.findall(strip_js_comments(script.read_text(encoding="utf-8")))
        if found:
            offenders[script.name] = found
    assert not offenders, f"user-facing text still hardcoded in the UI code: {offenders}"
