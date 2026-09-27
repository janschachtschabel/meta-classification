"""How the admin UI behaves when the happy path does not happen (audit FE-7, FE-9, FE-12…17).

Source rules, like `test_ui_a11y.py`: they cannot prove a screen reader announces a toast,
but they can prove the toast is dismissible, that a failure is not swallowed by a bare
`catch`, that a browser's own English does not reach a German user, and that a table with
one row per label is bounded. Each of these was a decision; a decision nothing argues for
comes back.
"""

import re
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "app" / "static" / "ui"
INDEX = (UI / "index.html").read_text(encoding="utf-8")
MODULES = {path.name: path.read_text(encoding="utf-8") for path in sorted(UI.glob("*.js"))}


def _js(name: str) -> str:
    return MODULES[name]


# --- FE-7: a dataset change must not race itself -------------------------------------------


@pytest.mark.parametrize(
    "module, handler",
    [("training.js", "loadDatasetColumns"), ("model-evaluate.js", "loadColumns")],
)
def test_reloading_a_datasets_columns_is_ordered_and_debounced(module, handler):
    """Arrow-keying a <select> fires `change` per keypress, and each one reads the whole CSV
    server-side. Without an ordering token a slow earlier answer lands after a newer one and
    the column picker then offers columns the selected dataset does not have."""
    source = _js(module)

    assert "latestOnly" in source, (
        f"{module}: {handler} still fires one full-CSV read per `change` event with no "
        "in-flight guard and no ordering token (train-status.js:pollInFlight is the pattern)"
    )


def test_the_ordering_helper_exists_once_and_drops_stale_answers():
    """One implementation, not one per caller — two copies drift, and this one decides
    whether a stale response may overwrite a newer one."""
    helper = _js("app.js")

    assert helper.count("function latestOnly") == 1, "latestOnly is not defined exactly once"
    # The token comparison is the whole point: without it the helper is just a debounce.
    assert re.search(r"[!=]==\s*seq\b|\bseq\s*[!=]==", helper), (
        "latestOnly does not compare a sequence number, so a stale answer still lands"
    )


# --- FE-9: the toast ----------------------------------------------------------------------


def test_the_toast_can_be_dismissed():
    """A 4 s auto-hide with no dismiss and no pause is a timing limit on reading (SC 2.2.1),
    and a long German error does not fit in 4 s."""
    assert "data-toast-dismiss" in _js("app.js"), "no dismiss control on the toast"


def test_a_failure_announces_itself_as_one():
    """`role="status"` is polite: it waits for a pause and can be dropped. Seven call sites
    routed errors through it. Errors get an assertive region of their own."""
    assert 'id="toast-alert"' in INDEX, "no assertive region for failures"
    assert re.search(r'id="toast-alert"[^>]*role="alert"', INDEX), (
        "#toast-alert is not role=alert, so a failure is still announced politely"
    )
    assert "function toastError" in _js("app.js"), "no toastError: errors still use toast()"


def test_a_failure_does_not_time_out_on_its_own():
    """A confirmation may disappear; a failure the user has to act on may not."""
    app = _js("app.js")
    start = app.index("function toastError")
    body = app[start:app.index("\nfunction ", start + 1)]

    assert "setTimeout" not in body, (
        "toastError schedules a hide: an error the user must act on can vanish unread"
    )


def test_a_second_message_does_not_silently_replace_the_first():
    """Two failures in a row showed one. The second overwrote the first before it was read."""
    app = _js("app.js")

    assert "insertAdjacentHTML" in app or "appendChild" in app, (
        "the toast still assigns textContent/innerHTML wholesale, so message N+1 erases N"
    )


def test_every_error_path_uses_the_assertive_channel():
    """A `toast(err…)` left anywhere puts a failure back in the polite region."""
    offenders = [
        f"{name}:{n}"
        for name, source in MODULES.items()
        for n, line in enumerate(source.splitlines(), 1)
        if re.search(r"(?<!Error)\btoast\(\s*err\b", line)
    ]

    assert not offenders, f"failures announced politely at: {offenders}"


# --- FE-12: a swallowed failure ----------------------------------------------------------


def test_the_share_listing_distinguishes_forbidden_from_broken():
    """Blanking the list is right for a readonly key's 403 and wrong for a 500: active links
    then vanish from the overview while remaining live and downloadable."""
    share = _js("share.js")

    assert re.search(r"catch\s*\(\s*err\s*\)[^\n]*\n?[^\n]*status\s*===?\s*403", share) or (
        "403" in share
    ), "share.js still blanks the list on any error, 500s included"
    assert not re.search(r"catch\s*\{\s*box\.innerHTML\s*=\s*\"\";\s*return;\s*\}", share), (
        "the bare catch that swallows every share-listing failure is still there"
    )


def test_a_dead_backend_stops_looking_like_a_running_job():
    """`catch { /* keep the last rendered state */ }` kept rendering "running · 40 % · ~12 min"
    forever after the server died, with nothing escalating."""
    status = _js("train-status.js")

    assert "consecutiveFailures" in status or "failedPolls" in status, (
        "train-status.js swallows every poll failure with no escalation, so a stale "
        "progress card is indistinguishable from a live one"
    )


# --- FE-13: the browser's own English is not a user-facing message -------------------------


def test_a_dropped_connection_is_reported_in_the_users_language():
    """`fetch` rejects with `TypeError: Failed to fetch` — untranslated browser internals,
    outside the string maps the i18n gate can see. Mapped once in the transport layer, so
    all eight display sites are covered by one change."""
    api = _js("api.js")

    assert "errors.network" in api, (
        "api.js does not translate a transport failure, so `TypeError: Failed to fetch` "
        "reaches the page verbatim"
    )
    for name in ("strings-de.js", "strings-en.js"):
        assert '"errors.network"' in _js(name), f"{name} has no errors.network"


# --- FE-14: the app's own validation is the one that runs ----------------------------------


def test_every_form_that_validates_itself_says_so():
    """A form with `required` fields and its own translated, role=alert checks lets the
    browser's native bubble fire first — in the BROWSER's locale — and the app's messages
    become dead code on that path. The login form already got this right."""
    forms = re.findall(r"<form\b[^>]*>", INDEX)
    missing = [f for f in forms if "novalidate" not in f]

    assert not missing, f"forms without novalidate: {missing}"


# --- FE-15: the clipboard is not always there ---------------------------------------------


def test_no_clipboard_write_is_unguarded():
    """`navigator.clipboard` is undefined outside a secure context, and this app documents
    that TLS terminates at a proxy — so over http:// the Copy click did nothing at all: no
    write, no toast, no error."""
    offenders = [
        f"{name}:{n}"
        for name, source in MODULES.items()
        if name != "app.js"
        for n, line in enumerate(source.splitlines(), 1)
        if "navigator.clipboard" in line
    ]

    assert not offenders, f"direct clipboard writes bypassing the guarded helper: {offenders}"


def test_the_copy_helper_reports_when_it_could_not_copy():
    helper = _js("app.js")

    assert "function copyText" in helper, "no shared copyText helper"
    assert "errors.clipboard" in helper, (
        "copyText fails silently: the user gets no toast and nothing on the clipboard"
    )


# --- FE-16: the per-label table is bounded ------------------------------------------------


def test_the_per_label_table_is_bounded_like_every_other_table():
    """This is the one view whose job is diagnosing a large label space, so it is the one
    that gets handed 300+ rows. query.js caps its table at 200 with a truncation notice."""
    detail = _js("model-detail.js")

    assert "LABEL_TABLE_LIMIT" in detail, "the per-label table renders every row unbounded"
    assert "labels.truncated" in detail, (
        "the table is capped with nothing saying so, which silently hides labels"
    )


def test_sorting_the_label_table_builds_its_collator_once():
    """`localeCompare` per comparison creates a collator per call — n log n of them on the
    view that has the most rows."""
    detail = _js("model-detail.js")

    assert "Intl.Collator" in detail, "still calling localeCompare per comparison"


# --- FE-17 -------------------------------------------------------------------------------


def test_the_signed_in_shell_has_a_top_level_heading():
    """The only <h1> lived in the hidden login section, so the whole app started at <h2>."""
    shell = INDEX[INDEX.index('id="view-app"'):]

    assert "<h1" in shell[:shell.index("</header>")], (
        "the app shell's brand is not an <h1>: the signed-in page has no level-1 heading"
    )


def test_the_page_declares_both_colour_schemes():
    """Without this the UA paints form controls and scrollbars light while the CSS paints
    the page dark."""
    assert re.search(r'<meta\s+name="color-scheme"\s+content="[^"]*dark', INDEX), (
        "no <meta name=color-scheme>, so UA-painted widgets ignore the dark theme"
    )


def test_a_scrollable_table_can_be_reached_by_keyboard():
    """An `overflow-x: auto` container is not focusable, so a keyboard user on Safari cannot
    scroll the label table sideways at all."""
    offenders = [
        f"{name}:{n}"
        for name, source in MODULES.items()
        for n, line in enumerate(source.splitlines(), 1)
        if 'class="table-wrap' in line and "tabindex" not in line
    ]

    assert not offenders, f"scroll containers a keyboard user cannot focus: {offenders}"


def test_a_csv_cell_cannot_become_a_formula():
    """A label beginning with `=`, `+`, `-` or `@` is a live formula when the download is
    opened in Excel or Calc. Quoting does not help — the parser consumes the quotes."""
    query = _js("query.js")
    neutraliser = re.search(r"const FORMULA_LEAD = /\^\[([^\]]+)\]/", query)
    cell = query[query.index("const csvCell"):][:400]

    assert neutraliser and set("=+-@") <= set(neutraliser.group(1)) and (
        "FORMULA_LEAD" in cell
    ), (
        "csvCell quotes but does not neutralise a leading =/+/-/@, so a crafted label "
        "executes when the download is opened"
    )


def test_html_escaping_is_written_once():
    """Two copies of the app's XSS escape is one that gets hardened and one that does not.
    They cannot simply be merged: pills.js renders during construction, before app.js has
    run, so the single definition has to load first."""
    defined_in = [
        name for name, source in MODULES.items()
        if re.search(r"^const esc\w* = \(s\)", source, re.M)
    ]

    assert defined_in == ["escape.js"], (
        f"esc is defined in {defined_in}: expected exactly one, in the first-loading module"
    )
    scripts = re.findall(r'<script src="([^"]+)"></script>', INDEX)
    assert scripts.index("escape.js") < scripts.index("pills.js"), (
        "escape.js must load before any module that renders at construction time"
    )


def test_the_model_picker_is_not_a_multi_select():
    """`<select multiple size="4">` needs Ctrl-click to pick a second model: undiscoverable
    with a mouse and, in several browsers, plainly unavailable from the keyboard. Carried
    over from the September audit as F4."""
    assert not re.search(r'id="query-models"[^>]*multiple', INDEX), (
        "the model picker is still a Ctrl-click multi-select"
    )
    assert 'id="query-models"' in INDEX, (
        "the picker's container id is gone: loadQueryTab needs it"
    )
