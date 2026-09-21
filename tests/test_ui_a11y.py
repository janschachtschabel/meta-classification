"""Accessibility invariants of the admin UI that a source rule can hold.

The automated part of WCAG is a minority of it, and none of this replaces tabbing through
the app with a screen reader. What it does replace is the drift: a live region added to a
container because the content inside it changes, an `aria-labelledby` pointing at an id
that only exists on the happy path. Those are decisions, and a decision that no longer has
to be argued for is one that comes back.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

UI = Path(__file__).parent.parent / "app" / "static" / "ui"
INDEX = (UI / "index.html").read_text(encoding="utf-8")

# Every element that may announce itself, and the reason it earns that. A live region is
# for a short STATUS MESSAGE the user did not navigate to; a container whose contents the
# app re-renders is not one, because a screen reader then reads the whole thing again on
# every tab switch, delete, upload and poll tick.
LIVE_REGIONS = {
    "query-status": "one line: how many texts were classified, or that none were",
    "train-announce": "state transitions only — the card itself re-renders every 2.5 s",
    "train-queue": "one line, and only while a run is actually queued",
    "train-preflight-announce": "the pre-flight's headline sentence, not its tables",
    "toast": "the status-message element itself",
}


class _Attributes(HTMLParser):
    """Every start tag's attributes, in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append({"__tag__": tag, **{k: v or "" for k, v in attrs}})


def _tags() -> list[dict[str, str]]:
    parser = _Attributes()
    parser.feed(INDEX)
    return parser.tags


def test_only_the_elements_that_earn_it_announce_themselves():
    """Adding a live region has to be a decision someone writes down.

    Five of them were containers — the model table, the dataset table, both share panels
    and the model-name preview — so switching to the Models tab re-announced every cell of
    every row, and typing a 20-character name queued 20 announcements that each
    interrupted the last. The training card carries a comment refusing exactly this; the
    rest of the file had drifted away from it.
    """
    announcing = {tag.get("id", "") for tag in _tags() if "aria-live" in tag}

    assert announcing == set(LIVE_REGIONS), (
        f"unexpected live regions: {sorted(announcing - set(LIVE_REGIONS))}; "
        f"missing: {sorted(set(LIVE_REGIONS) - announcing)}")


def test_nothing_is_both_a_live_region_and_a_description():
    """Two ways of saying the same thing, and they fight.

    `aria-describedby` is read when the described control takes focus — on demand, at the
    user's pace. `aria-live` interrupts. An element that is both announces its every
    change AND is read again on focus, which is how the model-name preview managed to
    speak twenty times while a name was being typed.
    """
    tags = _tags()
    described = {target for tag in tags for target in tag.get("aria-describedby", "").split()}
    announcing = {tag.get("id", "") for tag in tags if "aria-live" in tag}

    assert not (described & announcing), sorted(described & announcing)


# The two ids the markup references but does not contain: both name a <dialog> whose
# contents are rendered by its module, so the id arrives with the content. That the id
# arrives in EVERY state the dialog can be in is the other half, and it is
# test_each_dialog_names_itself_in_every_state_it_can_be_in below.
DIALOG_TITLES = {"model-detail-title": "model-detail.js",
                 "dataset-detail-title": "dataset-detail.js"}


def test_every_referenced_id_exists_in_the_markup():
    """A label or description pointing at nothing is a control with no accessible name,
    and it fails silently — the attribute is there, so a reviewer reads it as handled."""
    tags = _tags()
    ids = {tag["id"] for tag in tags if "id" in tag} | set(DIALOG_TITLES)
    dangling = {
        f'{attribute}="{target}"'
        for tag in tags
        for attribute in ("aria-describedby", "aria-labelledby", "aria-controls", "for")
        for target in tag.get(attribute, "").split()
        if target and target not in ids
    }

    assert not dangling, f"referenced ids that the markup never defines: {sorted(dangling)}"


def test_each_dialog_names_itself_in_every_state_it_can_be_in():
    """A dialog that is still loading, or that failed to load, is still a dialog.

    Both opened with a loading paragraph and named themselves through an id only the
    SUCCESS branch ever created, so `aria-labelledby` dangled for the whole life of a
    dialog whose request failed — announced as an unnamed dialog, with no way to tell
    which model it was about.

    Counted rather than searched: every write that replaces the dialog's own content is a
    state it can be in, so a fourth state added without a title makes the counts disagree.
    """
    for title, module in DIALOG_TITLES.items():
        source = (UI / module).read_text(encoding="utf-8")
        # `box.innerHTML` and the like fill a part of an already-named dialog; only these
        # two receivers replace the whole thing.
        states = len(re.findall(r"\b(?:dialog|frame)\.innerHTML\s*=", source))
        named = source.count(f'id="{title}"')

        assert states, f"{module}: found no dialog content writes — the receiver was renamed"
        assert named == states, (f"{module}: {states} states replace the dialog's content, "
                                 f"{named} of them carry id={title!r}")


def test_the_preview_of_the_planned_model_names_updates_without_announcing():
    """It is the description of the label-field input, which is the right mechanism: the
    names are there when a user asks for them, and silent while a name is being typed."""
    assert re.search(r'id="train-names-preview"[^>]*aria-live', INDEX) is None
    assert 'aria-describedby="train-names-preview"' in INDEX


def test_web_storage_is_never_touched_outside_a_guard():
    """Site data can be blocked outright — private windows, locked-down profiles — and
    then READING `sessionStorage` throws rather than returning null.

    `Api.getKey()` was the first thing `boot()` did, so on such a profile the throw
    escaped a `boot()` that had no `.catch`, both views stayed `hidden`, and the operator
    got a blank white page with nothing on it to act on. `i18n.js` had guarded the
    identical `localStorage` calls from the start, with a comment explaining this exact
    hazard; `api.js` had not.

    One line per access is this file's style, so "the line that touches storage also opens
    a try" is the whole rule.
    """
    unguarded = []
    for module in sorted(UI.glob("*.js")):
        for number, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            touches = "sessionStorage." in line or "localStorage." in line
            if touches and "try {" not in line:
                unguarded.append(f"{module.name}:{number}: {line.strip()}")

    assert not unguarded, "web storage touched outside a try/catch:\n" + "\n".join(unguarded)


def test_the_boot_sequence_cannot_end_in_a_blank_page():
    """Both views start `hidden` and `boot()` decides which one to show, so anything that
    escapes it leaves the page empty. The sign-in screen is the one state a person can act
    from, whatever went wrong."""
    app = (UI / "app.js").read_text(encoding="utf-8")

    assert "boot().catch(" in app, "boot() is invoked without a catch"
