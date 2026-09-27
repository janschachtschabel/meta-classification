"""The documented rate-limit buckets are the ones the code applies (audit DOC-5).

Derived from the decorators rather than read alongside them: the table in
`docs/configuration.md` is what an operator sizes a deployment from, and a route that quietly
joins a bucket makes that number wrong without anything failing.

Only the two rows that **enumerate** endpoints are checked. `APIV3_RATE_LIMIT_EXPORT` and
`APIV3_RATE_LIMIT_DEFAULT` describe their buckets by category on purpose ("import/export
endpoints", "fallback bucket"), and turning those into exhaustive lists would trade a
readable sentence for a list nobody reads and everybody has to maintain.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DOC = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")

# Resolver -> the env var whose row enumerates its endpoints.
_ENUMERATED = {
    "predict_limit": "APIV3_RATE_LIMIT_PREDICT",
    "train_limit": "APIV3_RATE_LIMIT_TRAIN",
}


def _routes_by_resolver() -> dict[str, set[str]]:
    """Every route path, grouped by the limiter resolver decorating it.

    Parsed per decorator block — the run of `@...` lines immediately above one `def`. A regex
    spanning from a `@router` line to the next `@limiter.limit` crosses route boundaries and
    attributes an undecorated route to whichever bucket appears further down the file.
    """
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "app" / "routes").glob("*.py")):
        route: str | None = None
        resolver: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("@router."):
                match = re.search(r'@router\.\w+\(\s*"([^"]+)"', stripped)
                if match:
                    route = match.group(1)
            elif stripped.startswith("@limiter.limit("):
                resolver = re.search(r"@limiter\.limit\((\w+)\)", stripped).group(1)
            elif stripped.startswith(("def ", "async def ")):
                if route and resolver:
                    found.setdefault(resolver, set()).add(route)
                route = resolver = None
        # A decorator run never closed by a `def` would be a syntax error, so nothing to flush.
    return found


@pytest.mark.parametrize("resolver", sorted(_ENUMERATED))
def test_every_route_in_an_enumerated_bucket_appears_in_its_row(resolver):
    routes = _routes_by_resolver().get(resolver, set())
    assert routes, f"no routes found for {resolver}: the parser stopped matching"

    row = next(
        line for line in CONFIG_DOC.splitlines()
        if line.startswith(f"| `{_ENUMERATED[resolver]}`")
    )
    # Parameter names are normalised away: the docs write `/share/{id}` where the route
    # template says `{share_id}`, and which word is inside the braces is not what this test
    # is about.
    def shape(text: str) -> str:
        return re.sub(r"\{[^}]*\}", "{}", text)

    missing = sorted(route for route in routes if shape(route) not in shape(row))

    assert not missing, (
        f"{_ENUMERATED[resolver]}'s row does not mention {missing} — an operator sizing this "
        "bucket would be sizing it for fewer endpoints than actually share it"
    )
