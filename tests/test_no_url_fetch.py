"""Import is upload-only, and the app has no way to fetch a URL at all.

CLAUDE.md states the rule ("model/dataset import via file upload only (no URL fetch)")
and both endpoints repeat it in their docstrings, but nothing failed if someone added a
`url=` form field and an HTTP client to satisfy a reasonable-sounding request. The rule
is a security boundary — a server that fetches a caller-supplied URL is an SSRF pivot
into whatever the deployment can reach — so it is asserted here rather than described.

Two independent assertions, because either alone can be satisfied while the property is
broken: the SOURCE may not import an outbound HTTP client, and the published CONTRACT
may not offer a URL to fetch.
"""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# Clients that reach the network. `urllib.parse` is fine — it only splits strings — so
# the check is against `urllib.request`, not the whole package.
OUTBOUND = {
    "httpx", "requests", "aiohttp", "urllib3", "http.client", "httplib2",
    "urllib.request", "ftplib", "telnetlib", "socket", "smtplib",
}

IMPORT_ENDPOINTS = ("/datasets/import", "/models/import")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_no_module_in_app_imports_an_outbound_http_client():
    """The structural half: with no client imported, no route CAN fetch a URL,
    whatever its parameters say."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(APP_DIR.rglob("*.py")):
        hits = {
            name for name in _imported_modules(path)
            if name in OUTBOUND or name.split(".")[0] in {"httpx", "requests", "aiohttp"}
        }
        if hits:
            offenders[str(path.relative_to(APP_DIR))] = hits
    assert not offenders, (
        f"outbound HTTP clients imported inside app/: {offenders}. Fetching a "
        f"caller-supplied URL server-side is an SSRF pivot; imports stay upload-only."
    )


def test_the_import_endpoints_take_a_file_and_offer_no_url():
    """The contract half: a client reads the schema, not the docstring. Both endpoints
    must ask for multipart file upload, and neither may expose a URL-ish field."""
    schema = TestClient(app).get("/openapi.json").json()
    for route in IMPORT_ENDPOINTS:
        body = schema["paths"][route]["post"]["requestBody"]["content"]
        assert "multipart/form-data" in body, f"{route} must take a file upload, got {list(body)}"
        # FastAPI publishes the form as a component reference, not inline.
        form = body["multipart/form-data"]["schema"]
        if "$ref" in form:
            form = schema["components"]["schemas"][form["$ref"].rsplit("/", 1)[-1]]
        properties = form.get("properties", {})
        assert "file" in properties, f"{route} has no file field: {sorted(properties)}"
        urlish = [name for name in properties if "url" in name.lower() or "uri" in name.lower()]
        assert not urlish, f"{route} offers {urlish}; import must not fetch anything"
        assert "application/json" not in body, (
            f"{route} also accepts JSON, which is how a url field usually arrives"
        )
