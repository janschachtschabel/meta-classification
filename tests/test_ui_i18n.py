"""The UI speaks German and English, and neither language may rot.

The admin UI has no build step and no test runner of its own (vanilla JS on
purpose), so the guards that a JS toolchain would give us live here instead:
the string maps are plain JSON inside a one-line assignment, which lets this
suite read them the way a translator does — as data.

Five things can go wrong with a hand-maintained i18n layer, and each has a test
below: a key exists in one language only, the markup or the code asks for a key
nobody defined, the map grows entries nothing uses, a key is assembled at
runtime where nothing can check it, and — the one that made D6 worth doing — a
string never made it into the map at all.

That last one is the hard one, because untranslated text does not fail: it just
quietly stays English in a German UI. A parser looks for it in the markup and
three nets look for it in the code. Each was verified to fire on a real example
before being trusted — a gate that has never caught anything is not a gate.
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

UI = Path(__file__).parent.parent / "app" / "static" / "ui"
INDEX = UI / "index.html"

# Attributes whose value a person reads. Each needs its own data-i18n-* twin, and
# test_the_applier_implements_every_hook_this_suite_guards keeps this list and
# i18n.js from drifting apart.
TRANSLATABLE_ATTRIBUTES = ("placeholder", "title", "aria-label", "alt", "label")

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
    """Remove /* */ and // comments so a comment cannot satisfy or trip a check.

    The ``//`` form needs whitespace or a line start in front of it, which
    leaves the ``//`` of a URL inside a string alone.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)(?:^|\s)//[^\n]*$", "", source)


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


_TEMPLATE_KEY_RE = re.compile(r"\bt\(\s*`")


def test_no_key_is_assembled_from_a_template():
    """``t(`x.${y}`)`` resolves at runtime, where no test can follow it.

    Every gate above works by reading keys out of the source, so a key that only
    exists once the code runs is a string neither language is checked for. The
    two places that would want one — the query submit button and the job states
    — carry explicit maps instead.
    """
    offenders = {p.name for p in ui_scripts()
                 if _TEMPLATE_KEY_RE.search(strip_js_comments(p.read_text(encoding="utf-8")))}
    assert not offenders, f"t() called with a template literal in: {sorted(offenders)}"


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


# A string carrying BOTH markup and a placeholder is the one shape where a
# forgotten esc() at the call site would put caller data into innerHTML. Five
# exist; the escaping of each was checked by hand. The list is here so that a
# sixth has to be added deliberately instead of appearing unnoticed.
MARKUP_WITH_PARAMETERS = {
    "feedback.modelSaid",              # labels: esc()d per entry in correctionHtml
    "modelDetail.fromVocabulary",      # vocabulary: esc()d at the call site
    "train.preflight.checkedAgainst",  # field: esc()d at the call site
    "train.preflight.keeps",           # numbers only
    "train.preflight.noEstimateFor",   # profile: esc()d at the call site
    "train.preflight.onProfile",       # profile: esc()d at the call site
    "train.preflight.underAMinuteOn",  # profile: esc()d at the call site
}


def test_only_reviewed_strings_mix_markup_with_a_placeholder():
    for language in ("de", "en"):
        mixed = {key for key, value in load_map(language).items()
                 if isinstance(value, str) and "<" in value and "{" in value}
        assert mixed == MARKUP_WITH_PARAMETERS, (
            f"{language}: the strings mixing markup and a placeholder changed — "
            f"newly mixed {sorted(mixed - MARKUP_WITH_PARAMETERS)}, "
            f"no longer mixed {sorted(MARKUP_WITH_PARAMETERS - mixed)}. "
            "Check that the call site escapes its parameter, then update the list."
        )


def test_the_applier_implements_every_hook_this_suite_guards():
    """The markup scanner demands a ``data-i18n-<attr>`` twin for every readable
    attribute; that demand is a lie unless i18n.js actually applies it."""
    applier = (UI / "i18n.js").read_text(encoding="utf-8")
    implemented = set(re.findall(r"\[data-i18n-([a-z-]+)\]", applier))
    missing = set(TRANSLATABLE_ATTRIBUTES) - implemented
    assert not missing, (
        f"i18n.js applies no {sorted(missing)} hook, so demanding one in the markup "
        "would let the attribute pass this suite and still never be translated"
    )


class _ProseScanner(HTMLParser):
    """Find text and human-readable attributes in the markup that no key covers.

    Text is "covered" when the element it sits in, or one of its ancestors,
    carries a ``data-i18n`` hook — an ancestor counts because a whole prose
    block (a help ``<details>``, a list) is translated as one string.

    Each frame remembers its tag and a closing tag unwinds down to the frame
    that opened it. HTML5 lets ``</li>``, ``</p>`` and friends be omitted and
    HTMLParser does not close them for us; popping one frame per closing tag
    would then shift the whole stack and leave "covered" stuck on for everything
    that follows — switching this scanner off precisely around a hooked block.
    """

    NO_CLOSING_TAG = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                      "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.skipping = 0            # inside <script>/<style>: not prose
        self.loose_text: list[str] = []
        self.loose_attributes: list[str] = []

    @property
    def covered(self) -> bool:
        return any(hooked for _, hooked in self.stack)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name in TRANSLATABLE_ATTRIBUTES:
            if attributes.get(name, "").strip() and f"data-i18n-{name}" not in attributes:
                self.loose_attributes.append(f"<{tag} {name}={attributes[name]!r}>")
        if tag in self.NO_CLOSING_TAG:
            return
        if tag in ("script", "style"):
            self.skipping += 1
        hooked = any(name in attributes for name in ("data-i18n", "data-i18n-html"))
        self.stack.append((tag, hooked))

    def handle_endtag(self, tag):
        if tag in self.NO_CLOSING_TAG:
            return
        if tag in ("script", "style") and self.skipping:
            self.skipping -= 1
        if not any(open_tag == tag for open_tag, _ in self.stack):
            return                   # stray closing tag: nothing of ours to unwind
        while self.stack.pop()[0] != tag:
            pass                     # drop the frames whose closing tag was omitted

    def handle_data(self, data):
        if self.skipping or self.covered or not data.strip():
            return
        self.loose_text.append(" ".join(data.split()))


def test_a_sentence_ending_in_an_abbreviation_gets_no_second_period():
    """German writes no second period after an abbreviation that ends a sentence, and the
    duration labels ARE abbreviations: "12 Min.", "1,5 Std.". The pre-flight appended its
    own period after the cost fragment and printed "12 Min..". The labels keep their
    period — the cost table shows them on their own — so the sentence has to ask whether
    one is already there instead of adding a second.

    Only the pre-flight embeds a duration in a sentence, so only training.js needs the
    helper. The other place that ends a built sentence with a period is the CSV summary
    in query.js, and it is safe for a reason this test keeps true: none of the fragments
    it joins ends in one. If that changes, the summary needs the same treatment.
    """
    german = load_map("de")
    assert german["common.minutes"].endswith("."), "the abbreviation keeps its period"
    assert german["common.hours"].endswith(".")

    training = (UI / "training.js").read_text(encoding="utf-8")
    assert "endSentence(" in training, "the pre-flight must not append a bare period"
    assert "}.</p>" not in training, "the bare period this test exists for"

    for language in ("de", "en"):
        strings = load_map(language)
        for key in ("query.csv.inputRows", "query.bulk.labelsAssigned",
                    "query.csv.rowsWithoutLabel"):
            for form in strings[key].values():
                assert not form.endswith("."), f"{key} ({language}) now needs endSentence"


def test_the_failure_announcement_does_not_double_the_period_of_its_message():
    """`trainStatus.announce.failed` embeds the run's error text, and those texts are
    sentences: "The training process ended without a result (exit code 137). Its log is
    in the server log." The string appended a period of its own, so the aria-live region
    a screen reader reads out ended in "..". The message carries the period; the
    announcement adds one only when it is missing."""
    for language in ("de", "en"):
        failed = load_map(language)["trainStatus.announce.failed"]
        assert "{message}" in failed, "the announcement quotes the run's error text"
        assert not failed.endswith("."), f"announce.failed ({language}) adds a period"

    status = (UI / "train-status.js").read_text(encoding="utf-8")
    assert "endSentence(" in status, "the announcement must close its own sentence"


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
    assert not scanner.stack, (
        f"index.html leaves {[tag for tag, _ in scanner.stack]} unclosed — the scanner "
        "then cannot tell what sits inside a translated block, and stops finding anything"
    )


def test_the_markup_scanner_survives_an_omitted_end_tag():
    """HTML5 lets ``</li>`` be left out, and the first version of this scanner
    then went blind for every sibling after the block it appeared in — silently,
    which is the worst way for a gate to fail."""
    scanner = _ProseScanner()
    scanner.feed('<div data-i18n-html="query.help"><ul><li>covered</ul></div>'
                 "<p>This sentence is hardcoded and nothing translates it.</p>")
    assert scanner.loose_text == ["This sentence is hardcoded and nothing translates it."]


# Three nets over the code, each aimed at a shape user-facing text actually takes
# here. None is a proof on its own; together they caught every string this change
# extracted, and each was verified to fire on a real example.
_NETS = {
    # A sentence: a capitalised word followed by a lowercase one.
    "sentence": re.compile(r"[\"'`]([A-ZÄÖÜ][a-zäöüß]+ [a-zäöüß]+[^\"'`]*)"),
    # Text between two tags in a template literal — a button label, a table cell.
    # Prose characters only, so that a regex literal or an HTML entity is not
    # mistaken for one; anything real goes through ${t(...)} and carries a $.
    "markup text": re.compile(
        r">([ 0-9A-Za-zÄÖÜäöüß.,:!?'’–—-]*[A-Za-zÄÖÜäöüß]{2}[ 0-9A-Za-zÄÖÜäöüß.,:!?'’–—-]*)<"),
    # A literal handed straight to something that shows it.
    "shown literal": re.compile(
        r"(?:toast|confirm)\(\s*[\"'][^\"']*[A-Za-zÄÖÜäöüß]{2}[^\"']*[\"']"
        r"|textContent\s*=\s*[\"'][^\"']*[A-Za-zÄÖÜäöüß]{2}[^\"']*[\"']"
        r"|message:\s*[\"'][^\"']*[A-Za-zÄÖÜäöüß]{2}[^\"']*[\"']"),
}


def test_no_untranslated_text_is_left_in_the_ui_code():
    offenders = {}
    for script in ui_scripts():
        source = strip_js_comments(script.read_text(encoding="utf-8"))
        for net, pattern in _NETS.items():
            found = pattern.findall(source)
            if found:
                offenders[f"{script.name} ({net})"] = found
    assert not offenders, f"user-facing text still hardcoded in the UI code: {offenders}"


def test_each_net_over_the_code_actually_catches_something():
    """A net nobody has seen fire is a net nobody should trust.

    These are the three shapes the D6 conversion actually found in this UI: a
    sentence in a quoted string, a label between two tags in a template
    literal, and a string handed straight to toast/confirm/textContent.
    """
    samples = {
        "sentence": '{ message: "Choose a model ZIP first." }',
        "markup text": '`<button class="small" data-archive="${n}">Archive</button>`',
        "shown literal": 'toast("Model archived.");',
    }
    for net, sample in samples.items():
        assert _NETS[net].search(sample), f"the {net!r} net no longer fires on {sample!r}"
