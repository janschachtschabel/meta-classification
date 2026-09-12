"""API-level tests via TestClient: auth, path-traversal, full train->predict flow.

Uses the offline TF-IDF 'fast' profile so no model is downloaded. Storage is
redirected to a temp dir via environment variables set before the app imports.
"""

import atexit
import csv
import io
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
        # Never the repo's own file: the suite trains several models and would
        # otherwise write real history entries on every run.
        "APIV3_JOB_HISTORY_FILE": str(_TMP / "job_history.jsonl"),
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
    # Explicit because the request default is 20 and the fixture holds 12 rows per
    # label — without this every label would be dropped and training would abort.
    "min_samples_per_label": 2,
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
    # 202 Accepted: the job was queued, not performed — the metrics arrive via
    # /train/status. Pinned so the async contract cannot quietly become a 200.
    assert started.status_code == 202, started.text
    # Exact key set: the TrainStartedResponse model silently DROPS any field it
    # doesn't declare — this guard turns a dropped contract field into a red test.
    assert set(started.json()) == {"status", "model_name", "profile", "status_url",
                                   "queue_position"}
    state = _wait_for_training()
    assert state["status"] == "completed", state
    return state


EVAL_BODY = {
    "dataset_name": "tiny.csv",
    "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
    "label_column": "properties.ccm:taxonid",
}


def test_an_editor_can_correct_a_prediction(trained_model):
    """The recognition rate improves with use, or it improves only when somebody
    produces a new export. A correction is what a person noticed, written down where
    the next training run can read it.

    Posting is readonly on purpose: correcting an answer is part of classifying, and
    the editors who spot the mistakes are exactly the ones without an admin key.
    """
    recorded = client.post("/feedback", headers=RO, json={
        "text": "Der Wiener Kongress von 1815",
        "model_name": "api_model",
        "predicted": ["uri:math"],
        "corrected": ["uri:hist"],
        "source": "ui",
    })
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["collected"] >= 1, "say how much has been gathered so far"

    exported = client.get("/feedback/export", headers=ADMIN)
    assert exported.status_code == 200, exported.text
    assert exported.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(exported.text), delimiter=";"))
    assert list(rows[0]) == ["text", "labels"], "the columns /train asks for"
    assert rows[-1]["labels"] == "uri:hist"


def test_feedback_is_bounded_at_the_trust_boundary(trained_model):
    """It is the one write a readonly key can make, so its size limits are the only
    thing standing between a key and an unbounded file."""
    assert client.post("/feedback", headers=RO, json={
        "text": "", "model_name": "api_model", "corrected": ["uri:hist"]}).status_code == 422
    assert client.post("/feedback", headers=RO, json={
        "text": "x", "model_name": "api_model",
        "corrected": [f"uri:{i}" for i in range(200)]}).status_code == 422
    assert client.post("/feedback", headers=RO, json={
        "text": "x", "model_name": "../escape", "corrected": ["uri:hist"]}).status_code == 400


def test_the_feedback_export_is_admin_only(trained_model):
    """Posting one correction is part of the job; walking off with every text an
    editor ever pasted is not."""
    assert client.get("/feedback/export", headers=RO).status_code == 403


def test_a_model_can_be_evaluated_on_a_dataset(trained_model):
    """"Model B beats model A" is only a statement if both were measured on the same
    rows. Until now that meant a script driving a running server, which nothing recorded
    and nobody could repeat.

    The result goes BESIDE the training metrics, never over them: those describe the run
    that produced the model and are the bundle's own account of itself.
    """
    started = client.post("/models/api_model/evaluate", json=EVAL_BODY, headers=ADMIN)
    assert started.status_code == 202, started.text
    assert _wait_for_training()["status"] == "completed"

    metadata = client.get("/models/api_model", headers=RO).json()["metadata"]
    assert metadata["metrics"]["f1_macro"] > 0.5, "the training metrics are untouched"

    evaluations = metadata["evaluations"]
    assert len(evaluations) == 1
    run = evaluations[0]
    assert run["dataset"] == "tiny.csv"
    assert run["metrics"]["f1_macro"] > 0.5
    assert run["n_rows"] > 0
    assert run["rows_without_a_known_label"] == 0
    assert run["evaluated_at"] and run["duration_seconds"] >= 0

    # The history has to say WHICH kind of run this was, or an evaluation reads as a
    # training that somehow produced no model.
    entry = next(e for e in client.get("/train/history", headers=RO).json()
                 if e["model_name"] == "api_model" and e["kind"] == "evaluation")
    assert entry["status"] == "completed"


def test_evaluating_twice_appends_rather_than_replaces(trained_model):
    """A model is evaluated on several datasets over its life; keeping only the newest
    would throw away exactly the comparison this exists for."""
    before = len(client.get("/models/api_model", headers=RO).json()["metadata"].get("evaluations", []))
    assert client.post("/models/api_model/evaluate", json=EVAL_BODY, headers=ADMIN).status_code == 202
    assert _wait_for_training()["status"] == "completed"

    after = client.get("/models/api_model", headers=RO).json()["metadata"]["evaluations"]
    assert len(after) == before + 1


def test_evaluating_an_unknown_model_or_dataset_is_refused(trained_model):
    assert client.post("/models/ghost/evaluate", json=EVAL_BODY, headers=ADMIN).status_code == 404
    assert client.post("/models/api_model/evaluate",
                       json={**EVAL_BODY, "dataset_name": "missing.csv"},
                       headers=ADMIN).status_code == 404
    assert client.post("/models/api_model/evaluate", json=EVAL_BODY, headers=RO).status_code == 403


def test_a_second_training_is_queued_instead_of_refused(trained_model):
    """Training five label fields used to need a browser tab kept open: the queue lived
    in the page, and closing it lost every run that had not started. A second POST is
    now accepted with its position, and the server runs it when the first is done.

    The training target is replaced with one that blocks on an Event — waiting for a
    real run to overlap another would be both slow and flaky.
    """
    import threading

    import app.routes.training as training_routes

    started, release = threading.Event(), threading.Event()

    def blocking(*_args, on_progress, should_stop, **_kwargs):
        started.set()
        release.wait(10)
        return {"model_name": "queued_first", "metrics": {"f1_macro": 0.5}, "n_labels": 1}

    original = training_routes.run_training
    training_routes.run_training = blocking
    try:
        first = client.post("/train", json={**TRAIN_BODY, "model_name": "queued_first"}, headers=ADMIN)
        assert first.status_code == 202, first.text
        assert first.json()["status"] == "started"
        assert first.json()["queue_position"] == 0
        assert started.wait(5)

        second = client.post("/train", json={**TRAIN_BODY, "model_name": "queued_second"}, headers=ADMIN)
        assert second.status_code == 202, second.text
        assert second.json()["status"] == "queued"
        assert second.json()["queue_position"] == 1

        # What is waiting is part of "what is going on here".
        assert client.get("/train/status", headers=RO).json()["queued"] == ["queued_second"]

        # The same name twice could only fail — /train refuses an existing model.
        again = client.post("/train", json={**TRAIN_BODY, "model_name": "queued_second"}, headers=ADMIN)
        assert again.status_code == 409, again.text
    finally:
        release.set()
        training_routes.run_training = original
        _wait_for_training()
        client.post("/train/stop", params={"hard": "true"}, headers=ADMIN)


def test_a_finished_run_is_in_the_history(trained_model):
    """Comparing two runs means opening two bundles and reading their metrics — and a
    run that FAILED leaves no bundle at all, so its reason lived only in whichever
    browser tab happened to be watching. The history is where an outcome outlives the
    tab, the restart and the deploy.
    """
    history = client.get("/train/history", headers=RO)
    assert history.status_code == 200, history.text
    entries = history.json()
    assert entries, "the run from the fixture is recorded"

    # By name AND kind: this module shares one server, other tests in it train, and a
    # model now carries evaluation entries too — "the newest entry for api_model" would
    # pin test order and, since B1, pick the wrong kind of run.
    entry = next(e for e in entries
                 if e["model_name"] == "api_model" and e["kind"] == "training")
    assert entry["status"] == "completed"
    assert entry["duration_seconds"] > 0
    assert entry["f1_macro"] > 0.5 and entry["n_labels"] >= 1
    # The request, so a run can be repeated or explained without guessing what it was.
    assert entry["request"]["dataset_name"] == "tiny.csv"
    assert entry["request"]["optimize_parameters"] == "fast"
    assert "info" not in entry["request"], "documentation is not a training parameter"


def test_the_history_records_a_run_that_failed(trained_model):
    """The failure case is the one with nothing else to inspect afterwards."""
    broken = {**TRAIN_BODY, "model_name": "history_failure", "min_samples_per_label": 9999}
    assert client.post("/train", json=broken, headers=ADMIN).status_code == 202
    assert _wait_for_training()["status"] == "error"

    entries = client.get("/train/history", headers=RO).json()
    entry = next(e for e in entries if e["model_name"] == "history_failure")
    assert entry["status"] == "error"
    assert entry["error"], "the reason survives the run"
    assert entry["f1_macro"] is None


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


def test_label_diagnostics_list_the_weakest_labels_first(trained_model):
    """"How good is this model" is one number; "where is it weak" is the question that
    decides whether to trust an answer. Per label: its F1, how many rows carry it, and
    the threshold serving applies — weakest first, because that is the end anyone
    reviewing a model looks at."""
    resp = client.get("/models/api_model/labels", headers=RO)
    assert resp.status_code == 200, resp.text
    labels = resp.json()

    info = client.get("/models/api_model", headers=RO).json()
    assert {entry["uri"] for entry in labels} == set(info["classes"])
    scores = [entry["f1"] for entry in labels]
    assert scores == sorted(scores), "weakest label first"
    for entry in labels:
        assert entry["label"] and entry["support"] >= 1
        assert 0.0 <= entry["f1"] <= 1.0

    assert client.get("/models/ghost/labels", headers=RO).status_code == 404


def test_label_diagnostics_report_a_threshold_only_where_serving_uses_one(trained_model):
    """tiny.csv is single-label, so the trained model decides by argmax and never
    reads a threshold. Reporting the global 0.5 there would describe a rule the model
    does not apply; a forced multilabel run does use its tuned per-label cuts."""
    single = client.get("/models/api_model/labels", headers=RO).json()
    assert all(entry["threshold"] is None for entry in single), "argmax reads no threshold"

    multi = {**TRAIN_BODY, "model_name": "multilabel_model", "task_type": "multilabel"}
    assert client.post("/train", json=multi, headers=ADMIN).status_code == 202
    assert _wait_for_training()["status"] == "completed"
    entries = client.get("/models/multilabel_model/labels", headers=RO).json()
    assert all(isinstance(entry["threshold"], float) for entry in entries)


def test_model_info_can_be_set_after_training_and_travels_with_the_bundle(trained_model):
    """The point of the info block is the moment a model is handed to someone else:
    the bundle records what was measured, but not who made it, what for, or where the
    data came from. It is editable after training — fixing a typo in an author name
    by retraining a model would be absurd — and it must survive export/import, or it
    would be lost exactly where it is needed."""
    info = {
        "author": "Redaktion WLO <redaktion@example.org>",
        "description": "Subject classifier for school material. Not for grading learners.",
        "data_source": "WLO prod export 2026-07-26",
        "license": "CC BY-SA 4.0",
    }
    stored = client.put("/models/api_model/info", json=info, headers=ADMIN)
    assert stored.status_code == 200, stored.text
    assert stored.json()["info"] == info
    assert client.get("/models/api_model", headers=RO).json()["metadata"]["info"] == info

    # PUT replaces: a second call with fewer fields leaves no remnants of the first.
    replaced = client.put("/models/api_model/info", json={"author": "Someone else"}, headers=ADMIN)
    assert replaced.status_code == 200
    assert replaced.json()["info"] == {"author": "Someone else"}

    client.put("/models/api_model/info", json=info, headers=ADMIN)
    export = client.post("/models/api_model/export", headers=ADMIN)
    files = {"file": ("m.zip", export.content, "application/zip")}
    assert client.post("/models/import", files=files,
                       data={"new_name": "api_shared"}, headers=ADMIN).status_code == 200
    assert client.get("/models/api_shared", headers=RO).json()["metadata"]["info"] == info


def test_train_stores_the_info_block_given_up_front():
    """Documentation supplied WITH the training request has to survive the route.
    `_REQ_KEYS` is an explicit allowlist of the fields that reach run_training, so a
    new body field is silently dropped until it is listed there — the bundle then
    comes out undocumented and nothing says why."""
    body = {
        **TRAIN_BODY,
        "model_name": "documented_model",
        "info": {"author": "Redaktion", "license": "CC BY-SA 4.0"},
    }
    assert client.post("/train", json=body, headers=ADMIN).status_code == 202
    assert _wait_for_training()["status"] == "completed"
    stored = client.get("/models/documented_model", headers=RO).json()["metadata"]["info"]
    assert stored == {"author": "Redaktion", "license": "CC BY-SA 4.0"}


def test_train_applies_the_weights_and_caps_the_request_asked_for():
    """The two levers a caller can set over the text itself — repeating a field in the
    training text, and the vocabulary caps — are validated by the schema and then have
    to REACH the run. `_REQ_KEYS` did not list them, so all three were accepted and
    dropped: a title weighting set in the form never reached a single model. The bundle
    records what was actually applied, which is where that shows."""
    body = {
        **TRAIN_BODY,
        "model_name": "weighted_model",
        "optimize_parameters": "auto",  # word AND char, so both caps are visible
        "text_column_weights": {"properties.cclom:title": 3},
        "max_word_features": 1_500,
        "max_char_features": 2_000,
    }
    assert client.post("/train", json=body, headers=ADMIN).status_code == 202
    assert _wait_for_training()["status"] == "completed"

    meta = client.get("/models/weighted_model", headers=RO).json()["metadata"]
    assert meta["text_column_weights"] == {"properties.cclom:title": 3}
    assert meta["tfidf"]["max_word_features"] == 1_500
    assert meta["tfidf"]["max_char_features"] == 2_000


def test_model_info_is_admin_only_and_bounded(trained_model):
    """Free text from a client that ends up rendered in the admin UI and shipped
    inside an exported bundle: bound it at the trust boundary, and keep writing it
    an admin action like every other model mutation."""
    assert client.put("/models/api_model/info", json={"author": "x"}, headers=RO).status_code == 403
    assert client.put("/models/ghost/info", json={"author": "x"}, headers=ADMIN).status_code == 404
    too_long = client.put("/models/api_model/info", json={"author": "x" * 500}, headers=ADMIN)
    assert too_long.status_code == 422, too_long.text
    unknown = client.put("/models/api_model/info", json={"nickname": "x"}, headers=ADMIN)
    assert unknown.status_code == 422, "an unknown field is a typo, not something to drop silently"


def test_import_garbage_zip_returns_400():
    """Uploading bytes that are not a zip yields a clean 400, not a 500."""
    files = {"file": ("evil.zip", b"this is not a zip archive", "application/zip")}
    response = client.post("/models/import", files=files, headers=ADMIN)
    assert response.status_code == 400


def test_import_rejected_while_training_same_name():
    """Importing a model whose training is currently running is refused (409):
    both operations would otherwise race on the same staging directory."""
    from app.jobs import job_runner

    job_runner.update(status="running", model_name="inflight")
    try:
        files = {"file": ("inflight.zip", b"irrelevant", "application/zip")}
        response = client.post("/models/import", files=files, headers=ADMIN)
        assert response.status_code == 409
    finally:
        job_runner.update(status="idle", model_name=None)


def test_import_rejected_while_a_same_name_training_is_queued(trained_model):
    """The running check above missed the queue: an import under a name that a
    queued training will save was accepted (reproduced before this change), and
    that training would later have replaced it."""
    import threading

    from app.jobs import job_runner

    bundle = client.post("/models/api_model/export", headers=ADMIN).content
    started, release = threading.Event(), threading.Event()

    def blocker(**_):
        started.set()
        release.wait(30)

    job_runner.submit(blocker, model_name="queue_blocker")
    try:
        assert started.wait(5)
        assert job_runner.submit(lambda **_: None, model_name="queued_import") == 1
        files = {"file": ("q.zip", bundle, "application/zip")}
        response = client.post("/models/import", files=files, data={"new_name": "queued_import"},
                               headers=ADMIN)
        assert response.status_code == 409
        assert "queued" in response.json()["detail"]
    finally:
        release.set()
        deadline = time.time() + 10
        while (job_runner.is_running() or job_runner.queued_names()) and time.time() < deadline:
            time.sleep(0.05)
    assert "queued_import" not in client.get("/models", headers=RO).text


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


def test_analyze_recommends_a_threshold_and_prices_the_run():
    """Analyzing a dataset is what you do BEFORE spending an afternoon on it, so the
    two numbers that decide the run belong in the answer: which
    `min_samples_per_label` fits this size, and roughly what each profile costs.

    Both are computed server-side on purpose — the recommendation already exists as
    `data.auto_min_samples` and the cost model has one measured anchor under it. A UI
    that re-derived either would drift from the thing that was actually measured.
    """
    body = client.post(
        "/datasets/analyze",
        json={
            "dataset_name": "tiny.csv",
            "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
            "label_column": "properties.ccm:taxonid",
        },
        headers=ADMIN,
    ).json()

    from app.data import auto_min_samples

    assert body["recommended_min_samples_per_label"] == auto_min_samples(body["total_samples"])
    # Paired with the threshold table so the recommendation can be READ as a decision:
    # "this value keeps N of M labels" is the sentence the UI has to be able to write.
    bucket = f"labels_with_{body['recommended_min_samples_per_label']}+_samples"
    assert bucket in body["label_threshold_analysis"]

    estimate = body["estimated_minutes"]
    assert set(estimate) == {"fast", "auto", "best"}
    assert all(value >= 0 for value in estimate.values())
    # Only non-decreasing here, deliberately: the fixture is 36 rows, where a model
    # anchored at 156 373 rounds all three profiles to 0.0 min — which is the truthful
    # output of a linear model at that size, not a bug. The STRICT ordering is a claim
    # about the model and is pinned where it means something (test_cost_estimate.py).
    assert estimate["fast"] <= estimate["auto"] <= estimate["best"]
    # The threads each estimate assumed, so a slow estimate explains itself: the deploy
    # fit's count under this server's CPU and memory budgets, never above the request.
    assert set(body["planned_head_fit_threads"]) == set(estimate)
    assert all(1 <= threads <= body["threads_requested"]
               for threads in body["planned_head_fit_threads"].values())


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


def test_gzipped_dataset_upload_list_download_and_analyze():
    """A .csv.gz dataset must work everywhere a .csv does.

    The WLO full exports are 126-195 MB gzipped against ~1.4 GB plain, and pandas reads the
    compressed form natively — so only the surrounding API stood in the way: the listing
    globbed `*.csv` (which never matches `*.csv.gz`), the upload demanded a `.csv` suffix,
    and the download announced `text/csv` for gzip bytes.
    """
    import gzip
    import io

    body = ("properties.cclom:title;properties.ccm:taxonid\n"
            + "".join(f"Titel {i};uri:x\n" for i in range(40)))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(body.encode("utf-8"))
    packed = buffer.getvalue()

    files = {"file": ("packed.csv.gz", packed, "application/gzip")}
    imported = client.post("/datasets/import", files=files, headers=ADMIN)
    assert imported.status_code == 200, imported.text
    assert imported.json()["dataset_name"] == "packed.csv.gz"

    listed = {d["name"]: d for d in client.get("/datasets", headers=RO).json()}
    assert "packed.csv.gz" in listed, "gzipped dataset missing from the listing"
    assert listed["packed.csv.gz"]["rows"] == 40, "row count read the compressed bytes"

    export = client.post("/datasets/packed.csv.gz/export", headers=ADMIN)
    assert export.status_code == 200
    assert "gzip" in export.headers["content-type"], export.headers["content-type"]
    assert gzip.decompress(export.content).decode("utf-8") == body, "download is not byte-faithful"

    # And the content is actually usable, not merely stored.
    info = client.get("/datasets/packed.csv.gz", headers=RO)
    assert info.status_code == 200, info.text
    assert "properties.ccm:taxonid" in info.json()["columns"]

    assert client.delete("/datasets/packed.csv.gz", headers=ADMIN).status_code == 200


def test_share_link_serves_gzipped_dataset_as_gzip():
    """A SHARED gzipped dataset must announce gzip too, not just the authenticated download.

    The share link is the path a recipient without an API key uses, so it is the one most
    likely opened in a browser — exactly where a `text/csv` header on gzip bytes leads to a
    silently decompressed file saved under its `.gz` name, which then opens nowhere.
    """
    import gzip
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(b"a;b\n1;2\n")
    files = {"file": ("shared.csv.gz", buffer.getvalue(), "application/gzip")}
    assert client.post("/datasets/import", files=files, headers=ADMIN).status_code == 200

    share = client.post("/datasets/shared.csv.gz/export",
                        json={"generate_share_url": True, "expires_hours": 1}, headers=ADMIN)
    fetched = client.get(f"/share/{share.json()['share_id']}")
    assert fetched.status_code == 200
    assert "gzip" in fetched.headers["content-type"], fetched.headers["content-type"]
    assert gzip.decompress(fetched.content) == b"a;b\n1;2\n"


def test_share_links_can_be_reviewed_and_revoked(trained_model):
    """Handing out a link is easy; taking it back was not possible through the API at
    all. Both are admin actions — the listing exposes the ids, which ARE the
    capability, so it must never answer a readonly key."""
    created = client.post("/models/api_model/export", headers=ADMIN,
                          json={"generate_share_url": True, "expires_hours": 1})
    share_id = created.json()["share_id"]
    assert client.get(f"/share/{share_id}").status_code == 200  # public, as designed

    assert client.get("/share", headers=RO).status_code == 403
    assert client.delete(f"/share/{share_id}", headers=RO).status_code == 403

    listed = client.get("/share", headers=ADMIN)
    assert listed.status_code == 200, listed.text
    entry = next(e for e in listed.json() if e["share_id"] == share_id)
    assert entry["kind"] == "model" and entry["name"] == "api_model"

    assert client.delete(f"/share/{share_id}", headers=ADMIN).status_code == 200
    assert client.get(f"/share/{share_id}").status_code == 404, "the link must stop working"
    assert share_id not in [e["share_id"] for e in client.get("/share", headers=ADMIN).json()]
    assert client.delete(f"/share/{share_id}", headers=ADMIN).status_code == 404


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


def test_a_share_link_never_serves_a_later_dataset_under_the_same_name():
    """Deleted and re-imported under its name, the old link handed out the NEW
    file -- reproduced before this change -- to someone nobody shared it with."""
    old = {"file": ("reused.csv", b"a;b\nold;1\n", "text/csv")}
    assert client.post("/datasets/import", files=old, headers=ADMIN).status_code == 200
    share = client.post("/datasets/reused.csv/export",
                        json={"generate_share_url": True, "expires_hours": 1}, headers=ADMIN)
    share_id = share.json()["share_id"]
    assert client.delete("/datasets/reused.csv", headers=ADMIN).status_code == 200

    new = {"file": ("reused.csv", b"a;b\nnew-and-private;2\n", "text/csv")}
    assert client.post("/datasets/import", files=new, headers=ADMIN).status_code == 200
    assert client.get(f"/share/{share_id}").status_code == 404
    assert share_id not in [e["share_id"] for e in client.get("/share", headers=ADMIN).json()]
    client.delete("/datasets/reused.csv", headers=ADMIN)


def test_a_share_link_never_serves_a_later_model_under_the_same_name(trained_model):
    bundle = client.post("/models/api_model/export", headers=ADMIN).content
    files = {"file": ("reused.zip", bundle, "application/zip")}
    assert client.post("/models/import", files=files, data={"new_name": "reused_model"},
                       headers=ADMIN).status_code == 200
    share = client.post("/models/reused_model/export",
                        json={"generate_share_url": True, "expires_hours": 1}, headers=ADMIN)
    share_id = share.json()["share_id"]
    assert client.delete("/models/reused_model", headers=ADMIN).status_code == 200

    assert client.post("/models/import", files=files, data={"new_name": "reused_model"},
                       headers=ADMIN).status_code == 200
    assert client.get(f"/share/{share_id}").status_code == 404
    client.delete("/models/reused_model", headers=ADMIN)


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
    # The training-config defaults a request would inherit when it omits them. The UI
    # pre-fills its form from these, so they have to be readable, not just documented.
    assert isinstance(body["default_text_column_weights"], dict)
    assert "default_min_samples_per_label" in body


def test_train_error_paths(trained_model):
    """/train rejects: unknown profile (400), missing dataset (404), existing model
    name (409). A second training while one runs is NO LONGER refused — it is queued
    (202), which is the change that let the browser tab stop shepherding runs; the
    remaining 409 on a busy server is a name already running or queued."""
    assert client.post("/train", json={**TRAIN_BODY, "optimize_parameters": "nope"},
                       headers=ADMIN).status_code == 400
    assert client.post("/train", json={**TRAIN_BODY, "dataset_name": "missing.csv",
                                       "model_name": "m2"},
                       headers=ADMIN).status_code == 404
    assert client.post("/train", json=TRAIN_BODY, headers=ADMIN).status_code == 409  # name exists

    from app.jobs import job_runner
    job_runner.update(status="running", model_name="other")
    try:
        queued = client.post("/train", json={**TRAIN_BODY, "model_name": "m3"}, headers=ADMIN)
        assert queued.status_code == 202, queued.text
        assert queued.json()["status"] == "queued"
        # The same name a second time is the case that stays a conflict.
        again = client.post("/train", json={**TRAIN_BODY, "model_name": "m3"}, headers=ADMIN)
        assert again.status_code == 409, again.text
    finally:
        # stop() clears the queue; without it the faked "running" state would leave a
        # real run waiting to be dispatched into the tests that follow.
        job_runner.stop()
        job_runner.update(status="idle", model_name=None)


def test_train_request_defaults_min_samples_per_label_to_20():
    """Dropping rare labels is a decision the user should SEE and be able to change,
    so the request declares 20 instead of quietly scaling it to the dataset size.
    Explicit null still asks for the size heuristic."""
    from app.schemas import TrainRequest

    body = {k: v for k, v in TRAIN_BODY.items() if k != "min_samples_per_label"}
    assert TrainRequest(**body).min_samples_per_label == 20
    assert TrainRequest(**body, min_samples_per_label=None).min_samples_per_label is None
    assert TrainRequest(**body, min_samples_per_label=5).min_samples_per_label == 5


def test_train_rejects_unusable_text_column_weights():
    """Bounds at the trust boundary. A weight for a column that is not being trained
    on is a typo the user would never notice (the field silently stays 1x), and an
    unbounded multiplier copies that column's text per row — cap it."""
    def post(weights: dict) -> int:
        return client.post(
            "/train",
            json={**TRAIN_BODY, "model_name": "weighted", "text_column_weights": weights},
            headers=ADMIN,
        ).status_code

    assert post({"properties.cclom:nope": 2}) == 422  # not among text_columns
    assert post({"properties.cclom:title": 0}) == 422  # 0 would drop the field
    assert post({"properties.cclom:title": 999}) == 422  # unbounded copies


def test_train_weight_check_stays_quiet_when_text_columns_itself_is_missing():
    """The cross-check needs text_columns. If THAT failed validation there is nothing
    to compare against, so claiming the weights name unknown columns would be an
    unfounded second error stacked on top of the real one."""
    from pydantic import ValidationError

    from app.schemas import TrainRequest

    with pytest.raises(ValidationError) as caught:
        TrainRequest(dataset_name="d.csv", model_name="m", label_column="l",
                     text_column_weights={"properties.cclom:title": 2})
    assert {err["loc"][0] for err in caught.value.errors()} == {"text_columns"}


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


def test_predict_label_f1_is_optional(trained_model):
    """include_label_f1=true attaches the label's training F1 to every prediction,
    so a caller can separate "confident here" from "reliable at all". Without the
    flag the field is absent (response unchanged)."""
    req = {"texts": ["Bruchrechnung und Gleichungen lösen"], "model_name": "api_model"}
    with_f1 = client.post("/predict", json={**req, "include_label_f1": True}, headers=RO)
    assert with_f1.status_code == 200, with_f1.text
    rows = with_f1.json()["results"][0]["predictions"]
    assert rows and all(0.0 <= p["label_f1"] <= 1.0 for p in rows)

    plain = client.post("/predict", json=req, headers=RO)
    assert all("label_f1" not in p for p in plain.json()["results"][0]["predictions"])


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


def test_predict_batch_is_an_alias_and_says_so(trained_model):
    """`/predict/batch` takes the same body, applies the same auth and rate limit and
    calls the same code as `/predict`, which already accepts a list of texts — its
    summary ("intended for larger batches") suggested a capability that does not
    exist. It stays for callers that use it, but the schema marks it deprecated so
    nobody picks it expecting different behaviour."""
    body = {"texts": ["Bruchrechnung üben", "Der Zweite Weltkrieg"], "model_name": "api_model"}
    plain = client.post("/predict", json=body, headers=RO)
    batch = client.post("/predict/batch", json=body, headers=RO)
    assert plain.status_code == batch.status_code == 200
    assert plain.json() == batch.json(), "an alias must not drift from what it aliases"

    schema = client.get("/openapi.json").json()["paths"]
    assert schema["/predict/batch"]["post"].get("deprecated") is True
    assert schema["/predict"]["post"].get("deprecated") is not True


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
    # The diagnostic endpoint always carries both reliability signals, unasked:
    # the empty-text baseline difference and the label's training F1.
    assert all("baseline_diff" in score for score in body["all_scores"].values())
    assert all("baseline_diff" in p for p in body["predictions"])
    assert all("label_f1" in score for score in body["all_scores"].values())
    assert all("label_f1" in p for p in body["predictions"])


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


def test_public_share_download_survives_a_bundle_with_a_malformed_metrics_document(trained_model):
    """`GET /share/{id}` is the one route with no API key — the id is the capability —
    and it packs the model card on the fly. metrics.json is not schema-validated on
    import, so a bundle whose metrics have the wrong types (an older exporter, a hand
    edit, a crafted upload) made that public route answer 500 instead of the file.

    The whole path is exercised, not the renderer alone: import -> share -> anonymous
    download, because that is how such a bundle actually reaches a stranger.
    """
    import io
    import json
    import zipfile

    export = client.post("/models/api_model/export", headers=ADMIN)
    assert export.status_code == 200

    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(export.content)) as source, \
            zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.namelist():
            if member in ("manifest.json", "README.md"):
                continue  # regenerated per export; a rewritten member must not match the old sums
            payload = source.read(member)
            if member == "metrics.json":
                payload = json.dumps({"n_samples": "many", "tfidf": [1, 2],
                                      "metrics": {"per_label_f1": {"uri:math": "high"}}}).encode()
            target.writestr(member, payload)

    files = {"file": ("bundle.zip", rebuilt.getvalue(), "application/zip")}
    imported = client.post("/models/import", files=files, data={"new_name": "odd_metrics"},
                           headers=ADMIN)
    assert imported.status_code == 200, imported.text

    shared = client.post("/models/odd_metrics/export", json={"generate_share_url": True},
                         headers=ADMIN)
    assert shared.status_code == 200, shared.text
    download = client.get(shared.json()["share_url"])  # no API key: that is the point
    assert download.status_code == 200, download.text
    assert download.headers["content-type"] == "application/zip"

    assert client.get("/models/odd_metrics/labels", headers=RO).status_code == 200
    assert client.get("/models/odd_metrics", headers=RO).status_code == 200


def test_import_rejects_a_bundle_whose_config_is_not_shaped_like_one(trained_model):
    """A bundle we cannot read is bad input (400), never a server fault (500).

    `_read_bundle` maps every shape error that way — but the container-label warning
    read `classes` *after* that block, so a `classes` that is not a list escaped it as
    a bare TypeError and the import answered 500 with a stack trace in the log.
    """
    import io
    import json
    import zipfile

    export = client.post("/models/api_model/export", headers=ADMIN)
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(export.content)) as source, \
            zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.namelist():
            if member in ("manifest.json", "README.md"):
                continue  # a rewritten member cannot match the exported checksums
            payload = source.read(member)
            if member == "config.json":
                payload = json.dumps({**json.loads(payload), "classes": 5}).encode()
            target.writestr(member, payload)

    files = {"file": ("bundle.zip", rebuilt.getvalue(), "application/zip")}
    refused = client.post("/models/import", files=files, data={"new_name": "bad_config"},
                          headers=ADMIN)
    assert refused.status_code == 400, refused.text
    assert "bad_config" not in client.get("/models", headers=RO).json()


def test_a_dataset_upload_that_dies_before_the_rename_leaves_nothing_behind(monkeypatch):
    """A dataset upload is capped at 200 MB and used to be assembled in memory first —
    the chunks, then the joined copy, so twice the file before a byte reached disk.

    It now spools straight into "<name>.part" and is renamed into place. The rename is
    what makes a dataset exist, so a failure before it must leave nothing that can be
    listed, inspected, downloaded or trained on — and no orphan on the volume either.
    """
    import app.routes.datasets as datasets_mod

    def _die(_src, _dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(datasets_mod.os, "replace", _die)
    files = {"file": ("crash.csv", b"title,label\nx,uri:math\n", "text/csv")}
    with pytest.raises(OSError):
        client.post("/datasets/import", files=files, headers=ADMIN)

    listed = [d["name"] for d in client.get("/datasets", headers=RO).json()]
    assert "crash.csv" not in listed
    assert client.get("/datasets/crash.csv", headers=RO).status_code == 404
    assert not list((_TMP / "data").glob("*.part")), "the partial upload is cleaned up"


def test_an_oversized_dataset_upload_is_refused_without_filling_the_disk(monkeypatch):
    """The cap has to bite while spooling, not after: the point of writing straight to
    disk is that the whole upload is never held anywhere, and a refusal that left the
    bytes on the volume would just move the problem."""
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)  # any byte is over the cap
    files = {"file": ("huge.csv", b"title,label\n" + b"x,uri:math\n" * 100, "text/csv")}
    response = client.post("/datasets/import", files=files, headers=ADMIN)
    assert response.status_code == 413, response.text

    assert "huge.csv" not in [d["name"] for d in client.get("/datasets", headers=RO).json()]
    assert not list((_TMP / "data").glob("*.part")), "the refused upload leaves no bytes"


CSV_BODY = (
    b"properties.cclom:title;properties.cclom:general_keyword;other\n"
    b"Bruchrechnung und Gleichungen loesen;Mathematik Brueche;x\n"
    b"Photosynthese der gruenen Pflanzen;Biologie Blatt Chlorophyll;y\n"
    b"Der Wiener Kongress von 1815;Geschichte Europa Restauration;z\n"
)


def test_a_whole_csv_can_be_classified_in_one_call(trained_model):
    """The editorial job is "classify these 500 new items", not one text. Doing that
    through /predict means the caller assembles each row's text — and how a text is
    assembled is part of what the model was fit on, so it is the one thing not to leave
    to the caller. The columns and their weights come out of the bundle.
    """
    files = {"file": ("items.csv", CSV_BODY, "text/csv")}
    response = client.post("/predict/csv", files=files,
                           data={"model_name": "api_model"}, headers=RO)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "items-predictions.csv" in response.headers["content-disposition"]

    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == ["row", "uri", "label", "confidence", "above_threshold"]
    assert {row[0] for row in rows[1:]} == {"0", "1", "2"}, "every input row is accounted for"
    assert rows[1][1] == "uri:math", "the first row is classified from its own text"
    assert 0.0 <= float(rows[1][3]) <= 1.0


def test_classifying_a_csv_refuses_a_multi_character_separator(trained_model):
    """pandas treats a multi-character `sep` as a REGEX and falls back to its python
    engine, so the separator becomes attacker-supplied pattern code running over the
    whole file. `/datasets/{name}` already refuses this for exactly that reason; this
    route dropped the guard — and it is reachable with a readonly key, on a
    single-worker server, where one upload can pin the CPU for hours.
    """
    files = {"file": ("items.csv", CSV_BODY, "text/csv")}
    refused = client.post("/predict/csv", files=files, headers=RO,
                          data={"model_name": "api_model", "separator": "(a+)+$"})
    assert refused.status_code == 400, refused.text
    assert "single character" in refused.text


def test_evaluating_refuses_a_dataset_name_that_escapes_the_data_directory(trained_model):
    """`/train` runs safe_name on the dataset name; this route did not, so the name went
    straight into a path join. An admin key could make the server read any parseable file
    on the box — and its column headers came back in the error and were written into the
    job history."""
    escaping = client.post("/models/api_model/evaluate", headers=ADMIN,
                           json={**EVAL_BODY, "dataset_name": "../secrets.csv"})
    assert escaping.status_code == 400, escaping.text
    absolute = client.post("/models/api_model/evaluate", headers=ADMIN,
                           json={**EVAL_BODY, "dataset_name": "C:/Windows/win.ini"})
    assert absolute.status_code == 400, absolute.text


def test_classifying_a_csv_refuses_a_file_without_the_trained_columns(trained_model):
    """A streaming response cannot report a failure — the status line is already 200 —
    so the header is checked while a 400 is still possible."""
    files = {"file": ("wrong.csv", b"headline;body\na;b\n", "text/csv")}
    response = client.post("/predict/csv", files=files,
                           data={"model_name": "api_model"}, headers=RO)
    assert response.status_code == 400, response.text
    assert "properties.cclom:title" in response.text, "say which column is missing"


def test_classifying_a_csv_reports_an_unknown_model_and_an_oversized_upload(trained_model):
    """Both failures belong to the request, not to the server, and both must be decided
    before a byte of CSV is streamed."""
    files = {"file": ("items.csv", CSV_BODY, "text/csv")}
    missing = client.post("/predict/csv", files=files,
                          data={"model_name": "ghost"}, headers=RO)
    assert missing.status_code == 404, missing.text

    from app.settings import get_settings

    settings = get_settings()
    original = settings.max_upload_mb
    try:
        settings.max_upload_mb = 0  # any byte is over the cap
        oversize = client.post("/predict/csv", files={"file": ("items.csv", CSV_BODY, "text/csv")},
                               data={"model_name": "api_model"}, headers=RO)
        assert oversize.status_code == 413, oversize.text
    finally:
        settings.max_upload_mb = original
    assert not list((_TMP / "data").glob(".predict-*")), "the spooled upload is cleaned up"
