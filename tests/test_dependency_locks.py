"""Supply-chain guard: the two lock files must agree.

``requirements.lock`` (human-maintained direct pins) is the input constraint for
the compiled ``requirements-hashes.lock`` (full transitive tree + sha256, used by
Docker/CI). If someone bumps the former and forgets to recompile the latter, the
shipped tree silently diverges from the tested one — this test turns that human
factor into a red gate.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", re.MULTILINE)


def _pins(filename: str) -> dict[str, str]:
    text = (ROOT / filename).read_text(encoding="utf-8")
    return {name.lower(): version for name, version in _PIN_RE.findall(text)}


def test_hashes_lock_matches_direct_pins():
    direct = _pins("requirements.lock")
    hashed = _pins("requirements-hashes.lock")
    assert len(direct) == 12, f"expected the 12 direct pins, parsed {sorted(direct)}"
    stale = {name: (version, hashed.get(name))
             for name, version in direct.items() if hashed.get(name) != version}
    assert not stale, (
        f"requirements-hashes.lock is stale for {stale} — recompile it "
        "(command in the requirements.lock header)."
    )


def test_hashes_lock_actually_carries_hashes():
    """The parity test alone would pass a recompile that dropped every hash;
    assert the hashes are really present so '--require-hashes' has teeth."""
    text = (ROOT / "requirements-hashes.lock").read_text(encoding="utf-8")
    pins = _PIN_RE.findall(text)
    assert pins, "no pins parsed from requirements-hashes.lock"
    assert text.count("--hash=sha256:") >= len(pins), (
        "requirements-hashes.lock carries fewer sha256 hashes than pinned "
        "packages — hash-pinning is not actually in effect."
    )
