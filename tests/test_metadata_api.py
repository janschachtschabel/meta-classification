"""`POST /metadata`: auth, the bounds at the trust boundary, and the response shape.

The generators themselves are covered by `tests/test_metadata_parity.py`, which checks them
against the pipeline they were ported from. What is asserted here is the endpoint around
them — that it is bounded, that it does not hold the single worker's event loop, and that a
text it can make nothing of is answered rather than refused.
"""

import ast
import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="apiv3-metadata-tests-"))
os.environ.update({
    "APIV3_AUTH_ENABLED": "true",
    "APIV3_API_KEY_ADMIN": "admin-key",
    "APIV3_API_KEY_READONLY": "ro-key",
    "APIV3_DATA_DIR": str(_TMP / "data"),
    "APIV3_MODELS_DIR": str(_TMP / "models"),
    "APIV3_JOB_HISTORY_FILE": str(_TMP / "job_history.jsonl"),
})

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import MetadataRequest  # noqa: E402

client = TestClient(app)
RO = {"X-API-Key": "ro-key"}

# Long enough to yield a heading, several description sentences and keywords.
SAMPLE = (
    "Die Fotosynthese\n"
    "Pflanzen stellen ihre Nahrung selbst her. Sie brauchen dafür Licht, Wasser und "
    "Kohlenstoffdioxid aus der Luft. In den Blättern sitzt der grüne Farbstoff Chlorophyll, "
    "der das Licht aufnimmt. Aus diesen Ausgangsstoffen entsteht Traubenzucker, und als "
    "Nebenprodukt geben die Pflanzen Sauerstoff ab. Ohne diesen Vorgang gäbe es auf der Erde "
    "keine Nahrungsketten.\n"
)


def _post(body: dict, headers: dict | None = RO):
    return client.post("/metadata", json=body, headers=headers)


def test_a_readonly_key_may_ask_for_metadata():
    """Deriving a title from a text the caller already has reads nothing and changes nothing,
    so it belongs to the same role as `/predict` — the editors who need it are the ones
    without an admin key."""
    response = _post({"texts": [SAMPLE]})

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["title"] == "Die Fotosynthese"
    assert result["description"].startswith("Pflanzen stellen ihre Nahrung selbst her.")
    assert "Fotosynthese" in result["keywords"]


def test_no_key_is_refused():
    assert _post({"texts": [SAMPLE]}, headers=None).status_code in (401, 403)


def test_results_come_back_one_per_text_in_order():
    """The caller matches answers to inputs by position, as with `/predict`."""
    response = _post({"texts": [SAMPLE, "Brüche addieren\nBrüche haben einen Zähler und "
                                        "einen Nenner. Gleichnamige Brüche werden addiert, "
                                        "indem man die Zähler addiert."]})

    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["title"] == "Die Fotosynthese"
    assert results[1]["title"] == "Brüche addieren"


def test_the_echoed_text_is_truncated_like_predict_does_it():
    """The echo is there to identify the row, not to return the input: a 100,000-character
    text in every one of 100 results would make the response an order of magnitude larger
    than the metadata it carries."""
    long_text = SAMPLE + "Ein weiterer Satz. " * 400
    echoed = _post({"texts": [long_text]}).json()["results"][0]["text"]

    assert len(echoed) <= 203 and echoed.endswith("...")


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"texts": []}, "an empty list has nothing to answer"),
        ({"texts": ["x" * 100_001]}, "one text past the per-text cap"),
        ({"texts": [SAMPLE] * 101}, "more texts than the batch cap"),
        # 99,996 characters each: under the per-text cap, so only the budget can refuse it.
        ({"texts": ["Satz. " * 16_666] * 11}, "more characters than one request's budget"),
        ({"texts": [SAMPLE], "n_keywords": 0}, "asking for no keywords"),
        ({"texts": [SAMPLE], "n_keywords": 101}, "asking for more keywords than exist"),
        ({"texts": [SAMPLE], "title_max": 4}, "a title budget nothing fits in"),
        ({"texts": [SAMPLE], "desc_max": 29}, "a description budget below one sentence"),
        ({"texts": [SAMPLE], "desc_max": 100_000}, "a description as long as the input"),
    ],
)
def test_the_bounds_hold_at_the_trust_boundary(body, why):
    """Every one of these is CPU the single worker would spend on a request that cannot be
    useful. 422, not a clamp: silently generating something other than what was asked for is
    how a caller ends up storing 500-character titles."""
    assert _post(body).status_code == 422, f"accepted {why}: {body.keys()}"


def test_the_budgets_in_the_request_are_the_ones_applied():
    """A caller with a 60-character title column has to be able to say so."""
    body = {"texts": [SAMPLE], "title_max": 20, "desc_max": 120, "n_keywords": 3}
    result = _post(body).json()["results"][0]

    assert len(result["title"]) <= 20
    assert len(result["description"]) <= 120
    assert len(result["keywords"]) <= 3


def test_a_text_that_yields_nothing_is_answered_not_refused():
    """"This text suggests no title" is a result. The caller asked what the text gives, and
    a 422 would say the request was wrong when it was the text that was thin."""
    response = _post({"texts": ["©"]})

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result == {"text": "©", "title": "", "description": "", "keywords": []}


def test_the_defaults_are_the_measured_ones():
    """The methods were measured at these budgets; a caller who sends none gets them."""
    from app.metadata import DEFAULT_SETTINGS

    empty = MetadataRequest(texts=["x"])
    assert (empty.title_max, empty.desc_max, empty.n_keywords) == (
        DEFAULT_SETTINGS.title_max, DEFAULT_SETTINGS.desc_max, DEFAULT_SETTINGS.n_keywords
    )


def test_generation_does_not_run_on_the_event_loop():
    """Measured at 12-22 ms per text and capped at 100 per request, so a full batch is ~2 s of
    pure CPU. The app is single-worker by contract: on the loop that is 2 s in which `/health`
    does not answer either.

    A source rule rather than a timing assertion, for the reason
    `tests/test_event_loop_is_not_blocked.py` gives: the property is "this work is handed to a
    thread", and that stays checkable whatever the machine is doing.
    """
    route = Path(__file__).resolve().parent.parent / "app" / "routes" / "metadata.py"
    tree = ast.parse(route.read_text(encoding="utf-8"))
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]

    assert handlers, "no async handler found — was the route renamed?"
    for handler in handlers:
        offloaded = any(
            isinstance(node, ast.Attribute) and node.attr == "to_thread"
            for node in ast.walk(handler)
        )
        assert offloaded, (
            f"{handler.name} generates on the event loop; hand the work to "
            f"asyncio.to_thread as /predict does"
        )


def test_the_endpoint_is_published_with_its_bounds():
    """The bounds are part of the contract: a client generates against the schema, and one
    that cannot see the batch cap discovers it as a 422 in production."""
    schema = client.get("/openapi.json").json()
    assert "/metadata" in schema["paths"], sorted(schema["paths"])

    body = schema["paths"]["/metadata"]["post"]["requestBody"]["content"]["application/json"]
    ref = body["schema"]["$ref"].rsplit("/", 1)[-1]
    texts = schema["components"]["schemas"][ref]["properties"]["texts"]
    assert texts["maxItems"] == 100
    assert texts["items"]["maxLength"] == 100_000


def test_markup_is_stripped_so_a_page_can_be_sent_as_it_is():
    """HTML in, clean fields out — the endpoint's side of `tests/test_metadata_markup.py`.

    This used to assert the opposite and say so: `data.clean_text` also collapses whitespace,
    and the line breaks it removes are what a title and a description are built from, so the
    contract asked callers to extract the text first. The patterns now live in `app.markup`
    and each pipeline composes them — flattening for the vectorizer, line-preserving here.
    """
    html = ("<h1>Bruchrechnung im Alltag</h1>\n"
            "<p>Brüche begegnen uns im Alltag häufig: beim Teilen einer Pizza oder in "
            "Rezepten.</p>")
    result = _post({"texts": [html]}).json()["results"][0]

    assert result["title"] == "Bruchrechnung im Alltag"
    assert "<" not in result["description"]
    assert "Bruchrechnung" in result["keywords"]



# --- M03 (audit 2026-09-30): what one request may cost -------------------------------------------


def test_a_request_is_held_to_a_character_budget_across_its_texts():
    """100 texts of 100,000 characters each passed every per-text bound, and at the documented
    caps one request cost about eight CPU-minutes (pysbd needs up to 259 ms per 4,000-character
    piece). The budget bounds the sum."""
    from app.schemas.metadata import MAX_REQUEST_CHARS

    within = MAX_REQUEST_CHARS // 100_000
    MetadataRequest(texts=["x" * 100_000] * within)
    with pytest.raises(ValueError, match="characters"):
        MetadataRequest(texts=["x" * 100_000] * (within + 1))


@pytest.fixture
def one_slot(monkeypatch, tmp_path):
    from app import concurrency
    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_MAX_CONCURRENT_METADATA", "1")
    get_settings.cache_clear()
    get_registry.cache_clear()
    concurrency.reset_slots()
    yield TestClient(create_app())
    get_settings.cache_clear()
    get_registry.cache_clear()
    concurrency.reset_slots()


def test_metadata_requests_beyond_the_concurrency_limit_are_refused_with_a_retry(one_slot):
    """Each batch holds a worker thread for its whole run, out of the pool every other route
    shares: eight long ones made `GET /models` wait 21 s. Refused, not queued -- the same rule
    as /predict/csv."""
    from app.concurrency import metadata_slots

    slots = metadata_slots()
    assert slots.capacity == 1
    assert slots.try_acquire()  # a batch already running
    try:
        busy = one_slot.post("/metadata", json={"texts": [SAMPLE]}, headers=RO)
    finally:
        slots.release()

    assert busy.status_code == 503, busy.text
    assert busy.headers["retry-after"]
    assert one_slot.post("/metadata", json={"texts": [SAMPLE]}, headers=RO).status_code == 200
    assert slots.in_use == 0, "the answered request kept its slot"
