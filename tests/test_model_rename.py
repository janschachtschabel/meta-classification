"""A model's name can be corrected, and a display name set beside it -- neither ever twice.

The name was the bundle's directory and nothing else: a typo at training or an import named
after its ZIP stayed for good. A rename moves the directory; the old name's share links are
revoked, because a link resolves BY NAME and would otherwise serve whatever model is later
given that name. The display name lives in the documentation block; no two models share a
name or a display name, in any letter case, across both.
"""

import json

import pytest
from fastapi.testclient import TestClient

ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}


def _bundle(models_dir, name, info=None):
    bundle = models_dir / name
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps({"format_version": 2, "classes": ["uri:a"],
                                                    "task_type": "multilabel"}), encoding="utf-8")
    metadata = {"n_labels": 1, **({"info": info} if info else {})}
    (bundle / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    models = tmp_path / "models"
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(models))
    monkeypatch.setenv("APIV3_SHARE_LINKS_FILE", str(tmp_path / "share_links.json"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))
    (tmp_path / "data").mkdir()
    _bundle(models, "fach_tpyo")
    _bundle(models, "stufe", info={"display_name": "Bildungsstufe"})

    from app.limiter import limiter
    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings
    from app.sharing import get_share_store

    for cached in (get_settings, get_registry, get_share_store):
        cached.cache_clear()
    limiter.reset()
    yield TestClient(create_app()), models
    for cached in (get_settings, get_registry, get_share_store):
        cached.cache_clear()


def _rename(client, old, new, headers=ADMIN):
    return client.post(f"/models/{old}/rename", headers=headers, json={"new_name": new})


# ------------------------------------------------------------------------------ rename ---


def test_a_rename_moves_the_model_and_revokes_the_old_names_links(setup):
    from app.sharing import get_share_store

    client, models = setup
    store = get_share_store()
    store.create("model", "fach_tpyo", 24)
    store.create("model", "fach_tpyo", 24)
    store.create("model", "stufe", 24)

    answer = _rename(client, "fach_tpyo", "fach_typo")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"name": "fach_typo", "revoked_links": 2}
    assert client.get("/models", headers=RO).json() == ["fach_typo", "stufe"]
    assert (models / "fach_typo" / "config.json").exists() and not (models / "fach_tpyo").exists()
    assert [link["name"] for link in store.list()] == ["stufe"], "another model's link stays"


def test_a_change_of_letter_case_alone_is_a_rename(setup):
    client, _ = setup

    answer = _rename(client, "fach_tpyo", "Fach_tpyo")

    assert answer.status_code == 200, answer.text
    assert client.get("/models", headers=RO).json() == ["Fach_tpyo", "stufe"]


@pytest.mark.parametrize(("new", "status"), [
    ("STUFE", 409),            # another model's name, in another case
    ("bildungsstufe", 409),    # another model's display name
    ("fach_tpyo", 400),        # nothing to do
    ("../x", 400),             # not a name
    ("fach.prebackup", 400),   # the repair's backups, which no route serves
])
def test_a_rename_refuses_what_would_collide_or_is_no_name(setup, new, status):
    client, models = setup

    answer = _rename(client, "fach_tpyo", new)

    assert answer.status_code == status, answer.text
    assert (models / "fach_tpyo" / "config.json").exists(), "nothing moved"


def test_a_rename_of_an_unknown_model_or_without_admin(setup):
    client, _ = setup

    assert _rename(client, "nope", "x").status_code == 404
    assert _rename(client, "fach_tpyo", "x", headers=RO).status_code == 403


def test_a_rename_waits_for_a_run_that_holds_either_name(setup, monkeypatch):
    from app.jobs import job_runner

    client, models = setup
    monkeypatch.setattr(job_runner, "queued_names", lambda: ["fach_typo"])

    answer = _rename(client, "fach_tpyo", "fach_typo")

    assert answer.status_code == 409, answer.text
    assert (models / "fach_tpyo").exists()


# ------------------------------------------------------------------------ display name ---


def test_a_display_name_is_stored_in_the_documentation(setup):
    client, _ = setup

    answer = client.put("/models/fach_tpyo/info", headers=ADMIN,
                        json={"display_name": "Fächer (Schule)", "author": "Redaktion"})

    assert answer.status_code == 200, answer.text
    info = client.get("/models/fach_tpyo", headers=RO).json()["metadata"]["info"]
    assert info == {"display_name": "Fächer (Schule)", "author": "Redaktion"}


@pytest.mark.parametrize("display", ["Bildungsstufe", "BILDUNGSSTUFE", "Stufe"])
def test_a_display_name_never_repeats_another_models_name_or_display_name(setup, display):
    client, _ = setup

    answer = client.put("/models/fach_tpyo/info", headers=ADMIN, json={"display_name": display})

    assert answer.status_code == 409, answer.text
    assert "stufe" in answer.json()["detail"]


def test_a_model_may_be_displayed_under_its_own_name(setup):
    client, _ = setup

    assert client.put("/models/stufe/info", headers=ADMIN, json={"display_name": "STUFE"}).status_code == 200
