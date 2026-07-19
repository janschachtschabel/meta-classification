"""API-level tests via TestClient: auth, path-traversal, full train->predict flow.

Uses the offline TF-IDF 'fast' profile so no model is downloaded. Storage is
redirected to a temp dir via environment variables set before the app imports.
"""

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
_TMP = Path(tempfile.mkdtemp())
# Module-level dir (holds trained skops bundles) — reclaim it when the test
# process exits instead of leaking several MB into the OS temp area per run.
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
(_TMP / "data").mkdir()
shutil.copy(FIXTURES / "tiny.csv", _TMP / "data" / "tiny.csv")

os.environ.update(
    {
        "APIV3_AUTH_ENABLED": "true",
        "APIV3_API_KEY_ADMIN": "admin-key",
        "APIV3_API_KEY_READONLY": "ro-key",
        "APIV3_DATA_DIR": str(_TMP / "data"),
        "APIV3_MODELS_DIR": str(_TMP / "models"),
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}

TRAIN_BODY = {
    "dataset_name": "tiny.csv",
    "model_name": "api_model",
    "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
    "label_column": "properties.ccm:taxonid",
    "optimize_parameters": "fast",
}


def _wait_for_training(timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/train/status", headers=RO).json()
        if state["status"] in ("completed", "error", "stopped"):
            return state
        time.sleep(0.5)
    raise TimeoutError("training did not finish in time")


@pytest.fixture(scope="module")
def trained_model() -> dict:
    """Train 'api_model' exactly once for the whole module and return the final
    training status. Tests declare this dependency instead of relying on file
    order (or re-training as a fallback)."""
    started = client.post("/train", json=TRAIN_BODY, headers=ADMIN)
    assert started.status_code == 200, started.text
    # Exact key set: the TrainStartedResponse model silently DROPS any field it
    # doesn't declare — this guard turns a dropped contract field into a red test.
    assert set(started.json()) == {"status", "model_name", "profile", "status_url"}
    state = _wait_for_training()
    assert state["status"] == "completed", state
    return state


def test_health_is_public():
    assert client.get("/health").status_code == 200


def test_auth_required_and_role_enforced():
    assert client.get("/models").status_code == 401  # no key
    assert client.post("/train", json=TRAIN_BODY, headers=RO).status_code == 403  # readonly on admin


def test_path_traversal_rejected():
    assert client.get("/models/e..vil", headers=RO).status_code == 400


def test_full_train_predict_flow(trained_model):
    assert trained_model["results"]["metrics"]["f1_macro"] > 0.5

    assert "api_model" in client.get("/models", headers=RO).json()

    info = client.get("/models/api_model", headers=RO).json()
    assert info["task_type"] == "multiclass"
    assert info["metadata"]["metrics"]["f1_macro"] > 0.5

    response = client.post(
        "/predict",
        json={"texts": ["Bruchrechnung und Gleichungen lösen"], "model_name": "api_model"},
        headers=RO,
    )
    assert response.status_code == 200
    predictions = response.json()["results"][0]["predictions"]
    assert predictions and predictions[0]["uri"] == "uri:math"


def test_export_import_roundtrip_via_api(trained_model):
    export = client.post("/models/api_model/export", headers=ADMIN)
    assert export.status_code == 200
    assert export.headers["content-type"] == "application/zip"

    files = {"file": ("api_model.zip", export.content, "application/zip")}
    imported = client.post("/models/import", files=files, data={"new_name": "api_copy"}, headers=ADMIN)
    assert imported.status_code == 200, imported.text
    assert "api_copy" in client.get("/models", headers=RO).json()


def test_import_garbage_zip_returns_400():
    """Uploading bytes that are not a zip yields a clean 400, not a 500."""
    files = {"file": ("evil.zip", b"this is not a zip archive", "application/zip")}
    response = client.post("/models/import", files=files, headers=ADMIN)
    assert response.status_code == 400


def test_import_rejected_while_training_same_name():
    """Importing a model whose training is currently running is refused (409):
    both operations would otherwise race on the same staging directory."""
    from app.jobs import training_job

    training_job.update(status="running", model_name="inflight")
    try:
        files = {"file": ("inflight.zip", b"irrelevant", "application/zip")}
        response = client.post("/models/import", files=files, headers=ADMIN)
        assert response.status_code == 409
    finally:
        training_job.update(status="idle", model_name=None)


def test_datasets_endpoints():
    listing = client.get("/datasets", headers=RO)
    assert listing.status_code == 200
    assert any(d["name"] == "tiny.csv" for d in listing.json())
    analyze = client.post(
        "/datasets/analyze",
        json={
            "dataset_name": "tiny.csv",
            "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
            "label_column": "properties.ccm:taxonid",
        },
        headers=ADMIN,
    )
    assert analyze.status_code == 200
    assert analyze.json()["total_samples"] > 0


def test_analyze_wrong_column_returns_400_with_message():
    """A typo in the label column is the most common user error: /datasets/analyze
    answers 400 with the crafted message (available columns), not a generic 500."""
    r = client.post(
        "/datasets/analyze",
        json={
            "dataset_name": "tiny.csv",
            "text_columns": ["properties.cclom:title"],
            "label_column": "properties.ccm:taxonid_TYPO",
        },
        headers=ADMIN,
    )
    assert r.status_code == 400, r.text
    assert "not found" in r.json()["detail"]


def test_dataset_validate_endpoint():
    """POST /datasets/{name}/validate takes ONE JSON object (aligned with
    /datasets/analyze) and reports column existence and quality warnings."""
    r = client.post(
        "/datasets/tiny.csv/validate",
        json={"text_columns": ["properties.cclom:title"],
              "label_column": "properties.ccm:taxonid"},
        headers=ADMIN,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    r404 = client.post(
        "/datasets/nope.csv/validate",
        json={"text_columns": ["y"], "label_column": "x"}, headers=ADMIN,
    )
    assert r404.status_code == 404

    # The pre-alignment contract (raw array body + query params) is gone: 422.
    legacy = client.post(
        "/datasets/tiny.csv/validate",
        params={"label_column": "properties.ccm:taxonid"},
        json=["properties.cclom:title"],
        headers=ADMIN,
    )
    assert legacy.status_code == 422


def test_dataset_import_export_delete_lifecycle():
    """Full dataset lifecycle: upload (only .csv, no duplicates), plain CSV
    download, delete (idempotence -> 404 on the second delete)."""
    files = {"file": ("extra.csv", b"col_a;col_b\n1;2\n3;4\n", "text/csv")}
    imported = client.post("/datasets/import", files=files, headers=ADMIN)
    assert imported.status_code == 200, imported.text
    assert imported.json()["dataset_name"] == "extra.csv"

    dupe = client.post("/datasets/import", files=files, headers=ADMIN)
    assert dupe.status_code == 409  # same name again -> conflict

    not_csv = {"file": ("evil.exe", b"MZ", "application/octet-stream")}
    assert client.post("/datasets/import", files=not_csv, headers=ADMIN).status_code == 400

    export = client.post("/datasets/extra.csv/export", headers=ADMIN)
    assert export.status_code == 200
    assert "text/csv" in export.headers["content-type"]
    assert b"col_a" in export.content

    assert client.delete("/datasets/extra.csv", headers=ADMIN).status_code == 200
    assert client.delete("/datasets/extra.csv", headers=ADMIN).status_code == 404


def test_share_link_for_deleted_dataset_returns_404():
    """A share link whose underlying dataset was deleted answers 404, not 500."""
    files = {"file": ("ephemeral.csv", b"a;b\n1;2\n", "text/csv")}
    assert client.post("/datasets/import", files=files, headers=ADMIN).status_code == 200
    share = client.post(
        "/datasets/ephemeral.csv/export",
        json={"generate_share_url": True, "expires_hours": 1},
        headers=ADMIN,
    )
    share_id = share.json()["share_id"]
    assert client.delete("/datasets/ephemeral.csv", headers=ADMIN).status_code == 200
    assert client.get(f"/share/{share_id}").status_code == 404


def test_model_share_link_and_delete(trained_model):
    """Model export as share link -> zip download via /share; deleting a model
    works once and 404s the second time (and kills its share link)."""
    export = client.post("/models/api_model/export", headers=ADMIN)
    files = {"file": ("todelete.zip", export.content, "application/zip")}
    assert client.post("/models/import", files=files, data={"new_name": "todelete"},
                       headers=ADMIN).status_code == 200

    share = client.post(
        "/models/todelete/export",
        json={"generate_share_url": True, "expires_hours": 1},
        headers=ADMIN,
    )
    assert share.status_code == 200, share.text
    share_id = share.json()["share_id"]
    dl = client.get(f"/share/{share_id}")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/zip"

    assert client.delete("/models/todelete", headers=ADMIN).status_code == 200
    assert client.delete("/models/todelete", headers=ADMIN).status_code == 404
    assert client.get(f"/share/{share_id}").status_code == 404  # model gone -> link dead


def test_train_profiles_endpoint():
    """GET /train/profiles exposes the quality/effort profiles with their knobs."""
    r = client.get("/train/profiles", headers=RO)
    assert r.status_code == 200
    body = r.json()
    names = {p["name"] for p in body["profiles"]}
    assert {"fast", "auto"} <= names
    fast = next(p for p in body["profiles"] if p["name"] == "fast")
    assert fast["max_char_features"] is None  # word-only profile has no char cap


def test_train_error_paths(trained_model):
    """/train rejects: unknown profile (400), missing dataset (404), existing
    model name (409), and a second training while one runs (409)."""
    assert client.post("/train", json={**TRAIN_BODY, "optimize_parameters": "nope"},
                       headers=ADMIN).status_code == 400
    assert client.post("/train", json={**TRAIN_BODY, "dataset_name": "missing.csv",
                                       "model_name": "m2"},
                       headers=ADMIN).status_code == 404
    assert client.post("/train", json=TRAIN_BODY, headers=ADMIN).status_code == 409  # name exists

    from app.jobs import training_job
    training_job.update(status="running", model_name="other")
    try:
        r = client.post("/train", json={**TRAIN_BODY, "model_name": "m3"}, headers=ADMIN)
        assert r.status_code == 409  # a job is already running
    finally:
        training_job.update(status="idle", model_name=None)


def test_train_stop_endpoint():
    """POST /train/stop responds for both modes; hard=true resets to idle."""
    soft = client.post("/train/stop", headers=ADMIN).json()
    assert soft["status"] == "stopping"
    assert set(soft) == {"status"}  # TrainStopResponse: exact contract key set
    assert client.post("/train/stop", params={"hard": "true"},
                       headers=ADMIN).json()["status"] == "idle"


def test_top_k_ranking_semantics(trained_model):
    """Explicit top_k = 'the K most probable labels' (ranking) for EVERY task
    type; without top_k a multiclass model returns the single argmax label."""
    ranked = client.post("/predict",
                         json={"texts": ["Bruchrechnung und Gleichungen"],
                               "model_name": "api_model", "top_k": 5},
                         headers=RO).json()
    preds = ranked["results"][0]["predictions"]
    assert len(preds) == 3                                   # tiny model has 3 labels
    assert preds[0]["uri"] == "uri:math"                     # best guess first
    assert preds[0]["confidence"] >= preds[-1]["confidence"] # ranked order

    default = client.post("/predict",
                          json={"texts": ["Bruchrechnung und Gleichungen"],
                                "model_name": "api_model"},
                          headers=RO).json()
    assert len(default["results"][0]["predictions"]) == 1    # argmax without top_k
    assert default["applied_settings"]["top_k"] == 1


def test_predict_baseline_diff_is_optional(trained_model):
    """include_baseline_diff=true attaches confidence-minus-empty-text-baseline to
    every prediction; without the flag the field is absent (response unchanged)."""
    req = {"texts": ["Bruchrechnung und Gleichungen lösen"], "model_name": "api_model"}
    with_diff = client.post("/predict", json={**req, "include_baseline_diff": True}, headers=RO)
    assert with_diff.status_code == 200, with_diff.text
    rows = with_diff.json()["results"][0]["predictions"]
    assert rows and all(isinstance(p["baseline_diff"], float) for p in rows)

    plain = client.post("/predict", json=req, headers=RO)
    assert all("baseline_diff" not in p for p in plain.json()["results"][0]["predictions"])


def test_predict_ignores_removed_use_auto_settings_field(trained_model):
    """The deprecated no-effect `use_auto_settings` field was removed from the
    schema. A client still sending it must be silently ignored (extra fields),
    never rejected with 422 — the removal stays backward-compatible."""
    r = client.post(
        "/predict",
        json={"texts": ["Bruchrechnung"], "model_name": "api_model", "use_auto_settings": True},
        headers=RO,
    )
    assert r.status_code == 200, r.text


def test_predict_multi_one_call_several_target_fields(trained_model):
    """/predict/multi classifies each text with several models (= target fields)
    in one call; each model applies its OWN tuned thresholds and evaluation stays
    per model. Unknown models 404; list bounds are enforced."""
    export = client.post("/models/api_model/export", headers=ADMIN)
    files = {"file": ("field2.zip", export.content, "application/zip")}
    assert client.post("/models/import", files=files, data={"new_name": "field2"},
                       headers=ADMIN).status_code == 200

    r = client.post(
        "/predict/multi",
        json={"texts": ["Bruchrechnung und Gleichungen lösen"],
              "model_names": ["api_model", "field2"]},
        headers=RO,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    row = body["results"][0]
    assert set(row["predictions_by_model"]) == {"api_model", "field2"}
    assert row["predictions_by_model"]["api_model"][0]["uri"] == "uri:math"
    assert set(body["applied_settings"]) == {"api_model", "field2"}  # per-model thresholds

    ghost = client.post("/predict/multi",
                        json={"texts": ["x"], "model_names": ["ghost"]}, headers=RO)
    assert ghost.status_code == 404
    assert client.post("/predict/multi", json={"texts": ["x"], "model_names": []},
                       headers=RO).status_code == 422
    assert client.post("/predict/multi",
                       json={"texts": ["x"], "model_names": ["m"] * 6},
                       headers=RO).status_code == 422


def test_predict_explain(trained_model):
    r = client.post(
        "/predict/explain",
        json={"text": "Bruchrechnung und Gleichungen lösen", "model_name": "api_model"},
        headers=RO,
    )
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body and "word_importance" in body and "all_scores" in body
    # The diagnostic endpoint always carries the empty-text baseline difference.
    assert all("baseline_diff" in score for score in body["all_scores"].values())
    assert all("baseline_diff" in p for p in body["predictions"])


def test_predict_rejects_empty_and_oversized_input():
    """Input bounds at the trust boundary: an empty list, an over-long batch, and
    an out-of-range top_k are rejected with 422 before any model work."""
    assert client.post("/predict", json={"texts": [], "model_name": "api_model"},
                       headers=RO).status_code == 422
    assert client.post("/predict", json={"texts": ["x"] * 1001, "model_name": "api_model"},
                       headers=RO).status_code == 422
    assert client.post("/predict", json={"texts": ["x"], "top_k": 999999, "model_name": "api_model"},
                       headers=RO).status_code == 422


def test_predict_on_unloadable_bundle_returns_422_not_500():
    """A model that exists on disk but cannot be safely loaded (corrupt skops)
    yields a clean 422, not a 500 leaking internals."""
    import json as _json

    broken = _TMP / "models" / "broken"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "config.json").write_text(_json.dumps({
        "backend_kind": "tfidf", "classes": ["a"], "task_type": "multiclass",
        "avg_labels": 1.0, "global_threshold": 0.5, "per_label_thresholds": {}, "uri_to_label": {},
    }), encoding="utf-8")
    (broken / "head.skops").write_bytes(b"not a real skops container")
    (broken / "vectorizer.skops").write_bytes(b"not a real skops container")

    r = client.post("/predict", json={"texts": ["x"], "model_name": "broken"}, headers=RO)
    assert r.status_code == 422, r.text


def test_unhandled_error_returns_clean_500(monkeypatch):
    """A route raising an unexpected error returns a sanitized 500 body via the
    global exception handler — no internal detail leaks."""
    from fastapi.testclient import TestClient as _TC

    from app import registry as _reg

    def _boom(self, name):
        raise RuntimeError("boom secret internals path=/etc/passwd")

    monkeypatch.setattr(_reg.Registry, "info", _boom)
    safe_client = _TC(app, raise_server_exceptions=False)
    r = safe_client.get("/models/api_model", headers=RO)
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error."}
    assert "boom secret" not in r.text


def test_share_link_is_bearer_capability_no_key_needed():
    """A share link is a bearer capability: the unguessable id (+ expiry) is the
    authorization, so GET /share/{id} needs no API key."""
    export = client.post(
        "/datasets/tiny.csv/export",
        json={"generate_share_url": True, "expires_hours": 1},
        headers=ADMIN,
    )
    assert export.status_code == 200, export.text
    share_id = export.json()["share_id"]

    r = client.get(f"/share/{share_id}")  # no X-API-Key header at all
    assert r.status_code == 200, r.text
    assert "text/csv" in r.headers["content-type"]


def test_admin_ui_served_same_origin():
    """The optional admin UI is served by the app itself (same-origin, no CORS):
    /ui/ answers with the HTML shell, assets resolve, and no API key is needed
    for the static page (data access happens client-side WITH the key)."""
    r = client.get("/ui/")  # no X-API-Key: the page itself is public like /docs
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="app"' in r.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200
    # Hardening headers apply to the UI too (middleware covers mounts).
    assert r.headers.get("x-content-type-options") == "nosniff"
    # UI assets must revalidate (no-cache): otherwise browsers keep executing a
    # stale app.js after an update (observed in the field).
    assert r.headers.get("cache-control") == "no-cache"
    assert client.get("/ui/app.js").headers.get("cache-control") == "no-cache"


def test_security_headers_on_every_response():
    """Baseline security headers are set on all responses (API hardening):
    no MIME sniffing, no framing, no referrer leakage."""
    for path in ("/health", "/models"):
        r = client.get(path, headers=RO)
        assert r.headers.get("x-content-type-options") == "nosniff", path
        assert r.headers.get("x-frame-options") == "DENY", path
        assert r.headers.get("referrer-policy") == "no-referrer", path


def test_metrics_endpoint_prometheus_format():
    """/metrics is public (ServiceMonitors cannot send custom auth headers) and
    exposes only operational gauges in the Prometheus text format."""
    r = client.get("/metrics")  # no API key
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    for gauge in ("apiv3_uptime_seconds", "apiv3_models_total",
                  "apiv3_models_in_memory", "apiv3_training_running"):
        assert gauge in body, gauge
    # Every non-comment line must be "name[{labels}] value" with a numeric value.
    for line in body.strip().splitlines():
        if not line.startswith("#"):
            float(line.rsplit(" ", 1)[1])


def test_cors_locked_down_by_default():
    """With an empty CORS allowlist (the default) no Access-Control-Allow-Origin
    header is emitted — browsers must not be granted cross-origin access."""
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_dataset_read_endpoints_are_rate_limited(monkeypatch):
    """GET /datasets and GET /datasets/{name} read whole CSV files for their row
    counts — throttled like the other heavy endpoints so a readonly key cannot
    hammer the single worker."""
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_export", "1/minute")
    for path in ("/datasets/tiny.csv", "/datasets"):
        first = client.get(path, headers=RO)
        assert first.status_code == 200, (path, first.text)
        second = client.get(path, headers=RO)
        assert second.status_code == 429, path


def test_share_download_is_rate_limited(monkeypatch):
    """GET /share/{id} is public (bearer capability), so it must be rate-limited —
    otherwise share ids could be brute-forced without any throttle."""
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_export", "1/minute")
    first = client.get("/share/definitely-not-a-real-share-id")
    assert first.status_code == 404
    second = client.get("/share/definitely-not-a-real-share-id")
    assert second.status_code == 429  # limit reached -> throttled, not another probe


def test_train_request_validator_edge_cases():
    """Trust-boundary validators: blank filter strings collapse to None (Swagger
    form fields), 'auto' task_type means auto-detect, unknown values are rejected."""
    from pydantic import ValidationError

    from app.schemas import TrainRequest

    assert TrainRequest(**{**TRAIN_BODY, "label_filter": "   "}).label_filter is None
    assert TrainRequest(**{**TRAIN_BODY, "task_type": "auto"}).task_type is None
    with pytest.raises(ValidationError):
        TrainRequest(**{**TRAIN_BODY, "task_type": "bogus"})


def test_train_request_cv_folds_validation_and_plumbing():
    """cv_folds: 1 is rejected (meaningless — neither split nor CV); valid values
    reach the training request dict the /train route builds via _REQ_KEYS."""
    import pytest
    from pydantic import ValidationError

    from app.routes import training as training_routes
    from app.schemas import TrainRequest

    with pytest.raises(ValidationError):
        TrainRequest(**{**TRAIN_BODY, "cv_folds": 1})
    with pytest.raises(ValidationError):
        TrainRequest(**{**TRAIN_BODY, "cv_folds": 21})

    body = TrainRequest(**{**TRAIN_BODY, "cv_folds": 5})
    req = {key: getattr(body, key) for key in training_routes._REQ_KEYS}
    assert req["cv_folds"] == 5
    assert TrainRequest(**TRAIN_BODY).cv_folds is None  # default: use config


def test_predict_on_model_vanished_after_check_returns_404(monkeypatch):
    """If a model passes exists() but its load raises FileNotFoundError (deleted in
    a TOCTOU window), predict returns a clean 404, not a 500."""
    from app import registry as reg_mod

    monkeypatch.setattr(reg_mod.Registry, "exists", lambda self, name: True)

    def _vanished(self, name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(reg_mod.Registry, "get", _vanished)
    r = client.post("/predict", json={"texts": ["x"], "model_name": "ghost"}, headers=RO)
    assert r.status_code == 404, r.text
