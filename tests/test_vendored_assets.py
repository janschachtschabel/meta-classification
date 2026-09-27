"""The vendored Swagger UI is inside a gate (audit DEP-3).

`app/static/swagger/swagger-ui-bundle.js` is 1.42 MB of third-party JavaScript served to every
`/docs` visitor. Because it is a committed file rather than a manifest entry, `pip-audit` never
sees it and no dependency bot watches it, and its README recorded no hash — so neither the
committed copy nor a future re-download could be verified against anything.

This does not make the file current; it makes it *checked*. The version it pins is recorded in
the README and is not known-vulnerable (both published advisories for this package affect
< 4.1.3), but updating it needs a download, which is a deliberate act for a human to take with
the hash below as the before-picture.
"""

import hashlib
import re
from pathlib import Path

import pytest

SWAGGER = Path(__file__).resolve().parents[1] / "app" / "static" / "swagger"


def _recorded_hashes() -> dict[str, str]:
    readme = (SWAGGER / "README.md").read_text(encoding="utf-8")
    return dict(re.findall(r"^\| `([^`]+)` \| `sha256:([0-9a-f]{64})` \|$", readme, re.MULTILINE))


@pytest.mark.parametrize("filename", ["swagger-ui-bundle.js", "swagger-ui.css", "swagger-init.js"])
def test_the_vendored_asset_matches_its_recorded_hash(filename):
    recorded = _recorded_hashes()

    assert filename in recorded, (
        f"{filename} is served to every /docs visitor but has no recorded hash in "
        "app/static/swagger/README.md — nothing can tell an intended update from an edit"
    )
    actual = hashlib.sha256((SWAGGER / filename).read_bytes()).hexdigest()
    assert actual == recorded[filename], (
        f"{filename} does not match the hash recorded in its README. If this was a deliberate "
        f"update, record the new hash (sha256:{actual}) and the new version there."
    )


def test_the_readme_records_a_hash_for_every_served_asset():
    """A hash table that quietly stops covering a file is the same gap again."""
    served = {f.name for f in SWAGGER.iterdir() if f.suffix in {".js", ".css"}}

    assert served <= set(_recorded_hashes()), (
        f"served without a recorded hash: {sorted(served - set(_recorded_hashes()))}"
    )
