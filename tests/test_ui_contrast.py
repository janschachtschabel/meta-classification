"""The colour tokens the admin UI is drawn from, measured rather than asserted.

`--border` shipped with the comment ">3:1 vs surface for field borders" and was 1.62:1 —
below the 3:1 SC 1.4.11 requires for the visual boundary of a control, in both themes, for
two years of reviews that read the comment and moved on. A number in a comment is a claim
nobody re-derives, so this file re-derives all of them: every ratio a comment states is
recomputed from the two colours it names, and the pairs a conformance level depends on are
checked against that level.

The arithmetic is WCAG 2.x relative luminance, which is short enough to write out and thus
needs no dependency. Cross-checked against the canonical pairs in
`test_the_measurement_agrees_with_the_published_reference_values`, because a contrast test
with a wrong contrast function is worse than none.
"""

import re
from pathlib import Path

import pytest

CSS = (Path(__file__).parent.parent / "app" / "static" / "ui" / "style.css").read_text(
    encoding="utf-8")

# `--name: #rrggbb;` with the comment that follows it on the same line, if any.
_DECLARATION = re.compile(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6});(?:[^\n]*?/\*(.*?)\*/)?")
# The shape every ratio claim in this file takes: "6.0:1 on accent".
_CLAIM = re.compile(r"([0-9]+\.[0-9]+)\s*:\s*1\s+on\s+([a-z-]+)")


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4


def _luminance(colour: str) -> float:
    value = colour.lstrip("#")
    red, green, blue = (int(value[at:at + 2], 16) / 255 for at in (0, 2, 4))
    return 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)


def contrast(one: str, other: str) -> float:
    """WCAG contrast ratio between two opaque colours, 1.0 to 21.0."""
    first, second = _luminance(one), _luminance(other)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _tokens() -> dict[str, dict[str, str]]:
    """The declared colours per theme: light is the `:root` block, dark the media query."""
    at = CSS.index("@media (prefers-color-scheme: dark)")
    return {theme: {name: value for name, value, _ in _DECLARATION.findall(block)}
            for theme, block in (("light", CSS[:at]), ("dark", CSS[at:]))}


def _claims() -> list[tuple[str, str, str, float, str]]:
    """Every (theme, token, colour, claimed ratio, other token) a comment states.

    All of them, not the first per comment: a token that names two backgrounds is making
    two claims, and the second is the one nobody would re-check.
    """
    at = CSS.index("@media (prefers-color-scheme: dark)")
    found = []
    for theme, block in (("light", CSS[:at]), ("dark", CSS[at:])):
        for name, value, comment in _DECLARATION.findall(block):
            for claim in _CLAIM.finditer(comment or ""):
                found.append((theme, name, value, float(claim.group(1)), claim.group(2)))
    return found


THEMES = ("light", "dark")


def test_the_measurement_agrees_with_the_published_reference_values():
    """The three pairs every contrast implementation is checked against."""
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)
    assert contrast("#595959", "#ffffff") == pytest.approx(7.0, abs=0.01)


def test_every_ratio_a_token_comment_claims_is_the_one_its_colours_produce():
    """The guard `--border` needed and did not have.

    A comment is the only place these ratios are written down, so a stale one is a
    conformance claim nobody can check without recomputing it — which is what this does.
    The comment must carry the measurement rounded to its own one decimal, rather than
    anything within a tolerance: "close enough" is how a number drifts.
    """
    tokens = _tokens()
    claims = _claims()
    assert claims, "the parser found no ratio claims at all — the comment format changed"

    wrong = []
    for theme, name, value, claimed, against in claims:
        other = tokens[theme].get(against)
        assert other, f"{theme} --{name} claims a ratio against undeclared --{against}"
        actual = contrast(value, other)
        if round(actual, 1) != claimed:
            wrong.append(f"{theme} --{name} {value}: comment says {claimed}:1 on "
                         f"--{against}, measured {actual:.2f}:1")
    assert not wrong, "token comments state ratios their colours do not produce:\n" + \
        "\n".join(wrong)


@pytest.mark.parametrize("theme", THEMES)
def test_a_controls_boundary_clears_the_three_to_one_sc_1_4_11_requires(theme):
    """SC 1.4.11: the visual boundary of a user-interface component needs 3:1 against what
    it sits on. A field is drawn on `--bg` inside a card on `--surface`, so both count."""
    tokens = _tokens()[theme]
    border = tokens["border-ui"]

    assert contrast(border, tokens["surface"]) >= 3.0
    assert contrast(border, tokens["bg"]) >= 3.0


@pytest.mark.parametrize("theme", THEMES)
def test_the_text_tokens_clear_the_four_point_five_to_one_sc_1_4_3_requires(theme):
    """Body-sized text on either of the two backgrounds it is ever drawn on."""
    tokens = _tokens()[theme]

    for name in ("text", "muted"):
        for background in ("surface", "bg"):
            ratio = contrast(tokens[name], tokens[background])
            assert ratio >= 4.5, f"{theme} --{name} on --{background} is {ratio:.2f}:1"


def test_every_control_boundary_is_drawn_with_the_control_token():
    """`--border` stays the hairline for cards, table rows and section rules; a control's
    own outline has to come from `--border-ui`, or the token exists and changes nothing.

    Matched on the selectors rather than on a count, so adding a control here fails loudly
    instead of silently inheriting the decorative line.
    """
    controls = (
        'input:not([type="checkbox"]):not([type="radio"]):not([type="file"]), select, textarea',
        ".pillbox",
        "button.ghost, button.tab",
    )
    for selector in controls:
        at = CSS.index(selector + " {") if selector + " {" in CSS else CSS.index(selector)
        block = CSS[at:CSS.index("}", at)]
        assert "var(--border-ui)" in block, (
            f"{selector!r} draws its boundary with something other than --border-ui")

def _composite(foreground: str, background: str, alpha: float) -> str:
    """The colour the compositor actually paints for `opacity: alpha` over `background`."""
    front = [int(foreground.lstrip("#")[at:at + 2], 16) for at in (0, 2, 4)]
    back = [int(background.lstrip("#")[at:at + 2], 16) for at in (0, 2, 4)]
    return "#" + "".join(
        f"{round(alpha * f + (1 - alpha) * b):02x}" for f, b in zip(front, back, strict=True)
    )


# Every `opacity` on a rule that contains text, and the token that text is drawn in.
# `button[disabled]` is deliberately absent: WCAG exempts an inactive control, and the app
# pairs the dimming with `cursor: progress` rather than relying on it to carry meaning.
_DIMMED_TEXT = re.compile(r"(?m)^([^\n{]*)\{[^}]*opacity:\s*(0?\.[0-9]+)")


def test_no_opacity_rule_dims_text_below_the_readable_ratio():
    """`opacity` composites the whole subtree, so a rule aimed at a row's decoration dims
    the badge inside it too. Measured, both rules that did this put their meaning-carrier
    under 4.5:1 — `--warn` at 3.00:1 and `--muted` at 2.78:1 in the light theme — while the
    comment beside one of them said the badge was what carried the meaning.

    Any new dimming of text has to prove itself here or use a quieter token instead.
    """
    exempt = {"button[disabled]"}
    offenders = []
    for selector, alpha in _DIMMED_TEXT.findall(CSS):
        if selector.strip() in exempt:
            continue
        for theme, tokens in _tokens().items():
            # The worst case over every token text in such a row can be drawn in.
            for name in ("text", "muted", "warn", "ok", "danger"):
                dimmed = _composite(tokens[name], tokens["surface"], float(alpha))
                ratio = contrast(dimmed, tokens["surface"])
                if ratio < 4.5:
                    offenders.append(
                        f"{selector.strip()} (opacity {alpha}) makes --{name} "
                        f"{ratio:.2f}:1 in the {theme} theme"
                    )
    assert not offenders, "opacity pushes text under 4.5:1:\n  " + "\n  ".join(offenders)


def test_the_two_rules_that_replaced_their_opacity_still_use_a_token():
    """The fix is a quieter COLOUR, not a smaller alpha. A future edit that reaches for
    `opacity` again is caught by the rule above; this catches the other direction, a
    hard-coded hex slipped in where the token was."""
    for selector in ("tr.stale td", ".pred.below-t"):
        at = CSS.index(selector + " {")
        block = CSS[at:CSS.index("}", at)]
        assert "var(--muted)" in block, f"{selector} no longer de-emphasises with a token"
        assert "opacity" not in block, f"{selector} is dimming with opacity again"
