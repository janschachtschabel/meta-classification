"""What the Query view renders, executed rather than pattern-matched.

`query.js` is a plain script with no imports and no DOM access at load time, so node
can run it with the handful of globals it uses stubbed out. That makes the answers it
builds testable as behaviour — which is what the near-miss follow-up needed: the
failure it can produce is a thrown card, and a regex over the source cannot see that.

Skipped where node is absent: the UI has no toolchain on purpose, so node is a
convenience here, not a build dependency.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

QUERY_JS = Path(__file__).parent.parent / "app" / "static" / "ui" / "query.js"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# The globals query.js expects from the page, reduced to what a render needs: `t`
# returns its key (so an assertion can name the string it wants), `esc` passes through,
# which keeps the probe about structure rather than escaping, and `fmtFixed` stands in
# for the locale-aware formatter in i18n.js.
_HARNESS = """
const fs = require("fs"), vm = require("vm");
const source = fs.readFileSync(process.argv[2], "utf8");
const sandbox = { t: (key) => key, esc: (s) => String(s), document: {}, console,
                  fmtFixed: (v, d) => Number(v).toFixed(d) };
try {
  // Appended to the source so it runs in the same scope: `const`/function
  // declarations of a script stay lexical and never reach the sandbox object.
  const html = vm.runInNewContext(source + "\\n" + process.argv[3], sandbox);
  console.log(JSON.stringify({ ok: true, html: String(html) }));
} catch (err) {
  console.log(JSON.stringify({ ok: false, error: String(err) }));
}
"""


def _render(expression: str, tmp_path: Path) -> dict:
    """Evaluate one expression against the real query.js and return its result."""
    harness = tmp_path / "harness.cjs"
    harness.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["node", str(harness), str(QUERY_JS), expression],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(done.stdout)


@needs_node
def test_a_model_named_like_an_object_property_still_renders(tmp_path):
    """The near-miss lookup is keyed by model name, and a model may legitimately be
    called `constructor` or `toString` (`safe_name` rejects paths, not property names).
    Read straight off the object, such a name returns something from Object.prototype
    instead of "nothing came back" — and the card then throws, replacing the whole
    answer (including the models that DID return) with an error."""
    result = _render('renderPredictions({ constructor: [] }, {})', tmp_path)

    assert result["ok"], result.get("error")
    assert "query.noLabelAboveThreshold" in result["html"]
    assert "query.nearestBelow" not in result["html"], "nothing was nearly matched"


@needs_node
def test_the_nearest_misses_are_rendered_under_the_decline_note(tmp_path):
    """The point of the follow-up: a model that reported nothing says what it almost
    said, below the note explaining why the list is empty."""
    nearest = '{ m: [{ uri: "u", label: "Chemie", confidence: 0.4, baseline_diff: 0.1 }] }'
    result = _render(f'renderPredictions({{ m: [] }}, {nearest})', tmp_path)

    assert result["ok"], result.get("error")
    html = result["html"]
    assert html.index("query.noLabelAboveThreshold") < html.index("query.nearestBelow")
    assert "Chemie" in html
