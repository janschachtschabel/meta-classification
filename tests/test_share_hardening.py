"""The share capability and the dataset import, hardened (audit SEC-9, SEC-10, SEC-11, SEC-12).

A share id *is* the authorization — there is no second factor — so it belongs to the same
class of secret as the API key, and the API key is already kept out of logs, URLs and error
bodies. These close the three places the id or the name it points at got looser treatment
than that: the access log, the filesystem join on the way back out, and the import that
publishes a name.
"""

import logging
import os
from pathlib import Path

import pytest
import yaml

from app.log_filters import RedactShareTokens

CHART = Path(__file__).resolve().parents[1] / "deploy" / "helm" / "classification-api"


# --- SEC-9: the id must not reach the access log -------------------------------------------


def _access_record(path: str) -> logging.LogRecord:
    """A record shaped like uvicorn's access log emits them."""
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:50000", "GET", path, "1.1", 200),
        None,
    )


def test_a_share_id_is_redacted_from_the_access_log():
    """uvicorn runs with access logging on, and the id is the whole capability: anyone with
    log access otherwise holds every outstanding link for up to seven days."""
    record = _access_record("/share/0123456789abcdef0123456789abcdef")

    assert RedactShareTokens().filter(record) is True
    assert "0123456789abcdef" not in record.getMessage()
    # Redacted, not dropped: which route was hit, and its status, are what the log is for.
    assert "/share/" in record.getMessage()


def test_a_query_string_on_the_share_path_is_redacted_too():
    record = _access_record("/share/deadbeefdeadbeef?download=1")

    RedactShareTokens().filter(record)
    assert "deadbeef" not in record.getMessage()


def test_the_share_listing_path_is_left_alone():
    """`GET /share` carries no id, and rewriting it would hide a real route from the log."""
    record = _access_record("/share")

    RedactShareTokens().filter(record)
    assert record.getMessage().endswith('"GET /share HTTP/1.1" 200')


def test_no_other_path_is_touched():
    record = _access_record("/models/faecher_300k_auto/export")

    RedactShareTokens().filter(record)
    assert "faecher_300k_auto" in record.getMessage()


def test_a_record_that_is_not_an_access_line_survives_the_filter():
    """The filter is installed on the root handlers, so it sees every record in the process."""
    record = logging.LogRecord("api_v3", logging.INFO, __file__, 1, "plain message", None, None)

    assert RedactShareTokens().filter(record) is True
    assert record.getMessage() == "plain message"


# --- SEC-10: the chart must not serve the key over plaintext --------------------------------


def test_the_insecure_ingress_path_is_opt_in():
    """The value contract only: `allowInsecure` exists, defaults to False, and the template
    names the guard. Whether a plaintext ingress is actually REFUSED is proven by rendering,
    in `tests/test_helm_chart.py` — a substring is present either way, so this test would
    hold with the condition inverted. It stays because it is the half that survives when
    helm is not installed."""
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    template = (CHART / "templates" / "ingress.yaml").read_text(encoding="utf-8")

    assert "allowInsecure" in values["ingress"], (
        "ingress.allowInsecure is missing: nothing distinguishes 'TLS is handled above me' "
        "from 'I forgot the TLS block'"
    )
    assert values["ingress"]["allowInsecure"] is False, "the insecure path must be opt-in"
    assert "fail" in template and "allowInsecure" in template, (
        "the ingress template no longer mentions the guard at all"
    )


# --- SEC-11: a name read back off disk is still validated ----------------------------------


def test_the_dataset_policy_alone_does_not_stop_a_traversal(tmp_path):
    """Why `download_shared` needs `safe_name` and not just the policy: `resolve_dataset`
    checks the SUFFIX and that the file exists, so a traversal ending in `.csv` gets past it.
    Pinned because it is the assumption the fix rests on — if `is_dataset_name` ever grows
    into a traversal check, the reason for the extra call has gone and this test says so."""
    from app.data import is_dataset_name

    assert is_dataset_name("../../../etc/passwd.csv") is True
    (tmp_path / "outside.csv").write_text("secret", encoding="utf-8")
    inner = tmp_path / "data"
    inner.mkdir()

    from app.data import resolve_dataset

    assert resolve_dataset(inner, "../outside.csv") == inner / "../outside.csv"


def test_safe_name_is_what_refuses_the_traversal():
    from fastapi import HTTPException

    from app.security import safe_name

    for name in ("../outside.csv", "../../etc/passwd.csv", "/abs/path.csv", ".hidden.csv"):
        with pytest.raises(HTTPException) as caught:
            safe_name(name, "dataset name")
        assert caught.value.status_code == 400, name


def test_the_public_download_route_validates_the_stored_name():
    """Asserted on the source: the traversal itself cannot be reached without forging the
    store, which is the point — the guard is there for when something else goes wrong."""
    source = (Path(__file__).resolve().parents[1] / "app" / "routes" / "share.py").read_text(
        encoding="utf-8")
    body = source[source.index("async def download_shared"):]

    assert 'safe_name(info["name"]' in body, (
        "download_shared does not run the stored name through safe_name; resolve_dataset "
        "alone only checks the suffix (see the test above)"
    )
    assert "resolve_dataset" in body, "download_shared bypasses the dataset policy"


def test_a_model_link_whose_stored_name_leaves_the_models_directory_is_a_404(monkeypatch, tmp_path):
    """S11 (audit 2026-09-30): the dataset branch re-checks the stored name (SEC-11), the
    model branch did not -- and `Registry._path` is a plain join. A store entry naming
    `../elsewhere/m` exported a bundle from beside the models directory to anyone holding
    the link. The store is a JSON file on the volume; the route is public."""
    import json
    from datetime import UTC, datetime, timedelta

    from fastapi.testclient import TestClient

    from app.profiles import Profile, TrainingConfig
    from app.registry import Registry, get_registry
    from app.settings import Settings, get_settings
    from app.sharing import get_share_store
    from app.training import run_training

    fixtures = Path(__file__).parent / "fixtures"
    elsewhere = Settings(data_dir=fixtures, models_dir=tmp_path / "elsewhere", auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0])
    config = TrainingConfig(default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
                            test_size=0.2, min_text_length=5, drop_duplicates=True,
                            min_samples_per_label=2)
    run_training({"dataset_name": "tiny.csv", "model_name": "m",
                  "text_columns": ["properties.cclom:title"], "label_column": "properties.ccm:taxonid",
                  "csv_separator": ";", "label_separator": ",", "label_filter": None},
                 elsewhere, config, profile, Registry(elsewhere.models_dir, 2),
                 on_progress=lambda **_: None, should_stop=lambda: False)
    (tmp_path / "models").mkdir()
    links = tmp_path / "share_links.json"
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    links.write_text(json.dumps({"forged": {"kind": "model", "name": "../elsewhere/m",
                                            "expires_at": expires}}), encoding="utf-8")
    for key, value in {"APIV3_AUTH_ENABLED": "true", "APIV3_API_KEY_ADMIN": "admin-key",
                       "APIV3_API_KEY_READONLY": "ro-key", "APIV3_DATA_DIR": str(tmp_path),
                       "APIV3_MODELS_DIR": str(tmp_path / "models"),
                       "APIV3_SHARE_LINKS_FILE": str(links)}.items():
        monkeypatch.setenv(key, value)
    for cached in (get_settings, get_registry, get_share_store):
        cached.cache_clear()
    from app.main import create_app

    try:
        response = TestClient(create_app()).get("/share/forged")
    finally:
        for cached in (get_settings, get_registry, get_share_store):
            cached.cache_clear()

    assert response.status_code == 404, f"{response.status_code}, {len(response.content)} bytes"
    assert not list((tmp_path / "models").glob(".export-*"))


# --- SEC-12: two imports of one name cannot clobber each other -----------------------------


def test_the_import_staging_name_is_unique_per_request():
    """`<name>.part` is deterministic, so two concurrent imports of one name wrote the same
    staging file and interleaved their bytes into whichever one won the rename."""
    source = (Path(__file__).resolve().parents[1] / "app" / "routes" / "datasets.py").read_text(
        encoding="utf-8")

    assert 'target.name + ".part"' not in source, "the staging name is still deterministic"
    assert "mkstemp" in source, "no unique staging name for an upload"


def test_publishing_an_import_cannot_overwrite_an_existing_dataset(tmp_path):
    """The 409 check ran before the upload and `os.replace` overwrites silently, so a name
    created during the upload was replaced by it regardless."""
    from app.routes.datasets import _publish_new_dataset

    staging = tmp_path / ".import-abc.part"
    staging.write_text("new", encoding="utf-8")
    target = tmp_path / "taken.csv"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _publish_new_dataset(staging, target)

    assert target.read_text(encoding="utf-8") == "original", "the existing dataset was replaced"
    assert not staging.exists(), "the staging file was left behind"


def test_publishing_an_import_to_a_free_name_works(tmp_path):
    from app.routes.datasets import _publish_new_dataset

    staging = tmp_path / ".import-abc.part"
    staging.write_text("payload", encoding="utf-8")
    target = tmp_path / "fresh.csv"

    _publish_new_dataset(staging, target)

    assert target.read_text(encoding="utf-8") == "payload"
    assert not staging.exists()
    # And nothing the sweep would have to clean is left in the directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fresh.csv"]


def test_the_staging_file_is_invisible_to_every_route_that_names_a_dataset(tmp_path):
    """Whatever the unique name is, it must still be unnameable: the listing globs the CSV
    suffixes, and `safe_name` rejects a leading dot."""
    from app.routes.datasets import _staging_path

    path = _staging_path(tmp_path, "data.csv")
    try:
        assert path.name.startswith("."), path.name
        assert path.name.endswith(".part"), path.name
        assert not path.name.endswith((".csv", ".csv.gz"))
    finally:
        os.unlink(path)
