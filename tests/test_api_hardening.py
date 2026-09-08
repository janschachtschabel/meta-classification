"""Hardening tests (audit 2026-07-16): unified error envelope, CORS credential
guard, separator validation, and the now-wired default rate limit. Each builds a
fresh app so the process-global limiter/settings state never leaks between cases.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _fresh_client(monkeypatch, tmp_path, **env) -> TestClient:
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    # The registry is an lru_cache singleton bound to the settings active at its
    # FIRST call — without this, a fresh app still serves models from the
    # previous test file's directory (cross-file pollution).
    get_registry.cache_clear()
    from app.limiter import limiter
    from app.main import create_app

    limiter.reset()
    return TestClient(create_app())


RO = {"X-API-Key": "ro-key"}


def test_rate_limit_429_uses_detail_envelope(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path, APIV3_RATE_LIMIT_EXPORT="1/minute")
    first = client.get("/datasets", headers=RO)
    assert first.status_code == 200, first.text
    second = client.get("/datasets", headers=RO)
    assert second.status_code == 429
    # Same {"detail": ...} shape as every other error (not slowapi's {"error": ...}).
    assert "detail" in second.json()
    assert "error" not in second.json()


def test_dataset_info_rejects_multichar_separator(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "d.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    resp = client.get("/datasets/d.csv", params={"separator": ";;"}, headers=RO)
    assert resp.status_code == 400
    assert "single character" in resp.json()["detail"]


def test_dataset_info_empty_csv_returns_400_not_500(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "empty.csv").write_bytes(b"")
    resp = client.get("/datasets/empty.csv", headers=RO)
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_dataset_info_reads_cp1252_file(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "de.csv").write_bytes("titel;wert\nGröße;1\n".encode("cp1252"))
    resp = client.get("/datasets/de.csv", headers=RO)
    assert resp.status_code == 200, resp.text
    assert resp.json()["columns"] == ["titel", "wert"]


ADMIN = {"X-API-Key": "admin-key"}


def test_validate_maps_malformed_csv_to_400_not_500(monkeypatch, tmp_path):
    """validate must map TrainingInputError to a crafted 400 like its siblings
    (analyze, dataset_info) — not fall through to the sanitized 500 handler."""
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "empty.csv").write_bytes(b"")
    resp = client.post("/datasets/empty.csv/validate",
                       json={"text_columns": ["title"], "label_column": "lab"}, headers=ADMIN)
    assert resp.status_code == 400, resp.text
    assert "detail" in resp.json()


def test_multichar_separator_rejected_on_all_endpoints(monkeypatch, tmp_path):
    """The single-char separator guard must hold at EVERY separator input (pandas
    treats a multi-char sep as a regex -> ReDoS), not only on dataset_info."""
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "d.csv").write_text("a;b\n1;2\n", encoding="utf-8")

    train = client.post("/train", json={
        "dataset_name": "d.csv", "model_name": "m", "text_columns": ["a"],
        "label_column": "b", "csv_separator": "(x+)+", "optimize_parameters": "fast",
    }, headers=ADMIN)
    assert train.status_code == 422, train.text

    analyze = client.post("/datasets/analyze", json={
        "dataset_name": "d.csv", "text_columns": ["a"], "label_column": "b",
        "csv_separator": "(x+)+",
    }, headers=ADMIN)
    assert analyze.status_code == 422, analyze.text

    validate = client.post("/datasets/d.csv/validate",
                           json={"text_columns": ["a"], "label_column": "b",
                                 "csv_separator": "(x+)+"}, headers=ADMIN)
    assert validate.status_code == 422, validate.text


def test_unknown_profile_detail_has_no_stray_quotes(monkeypatch, tmp_path):
    """str(KeyError) reprs its message -> the 400 detail arrived wrapped in
    literal quotes. The detail must start with the message itself."""
    client = _fresh_client(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "d.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    resp = client.post("/train", json={
        "dataset_name": "d.csv", "model_name": "m", "text_columns": ["a"],
        "label_column": "b", "optimize_parameters": "does-not-exist",
    }, headers=ADMIN)
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("Unknown profile"), resp.json()["detail"]


def test_predict_on_corrupt_config_json_returns_422_not_500(monkeypatch, tmp_path):
    """model_io promises 'any bundle we cannot safely load' -> UnsafeModelError;
    a corrupt config.json must surface as 422 like a corrupt skops file."""
    client = _fresh_client(monkeypatch, tmp_path)
    broken = tmp_path / "models" / "broken_cfg"
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("not json at all {", encoding="utf-8")
    (broken / "head.skops").write_bytes(b"junk")
    (broken / "vectorizer.skops").write_bytes(b"junk")
    resp = client.post("/predict", json={"texts": ["x"], "model_name": "broken_cfg"},
                       headers=RO)
    assert resp.status_code == 422, resp.text


def test_ui_sources_contain_no_inline_event_handlers():
    """The UI runs under `default-src 'self'` WITHOUT 'unsafe-inline' — any
    inline on*= handler (also when injected via innerHTML) is silently blocked
    by the CSP. Pin the invariant at the source level."""
    import re
    from pathlib import Path

    ui_dir = Path(__file__).parent.parent / "app" / "static" / "ui"
    # HTML inline handlers only: lowercase attribute name directly assigned a
    # quoted value (onfocus="..."). Deliberately case-sensitive and quote-anchored
    # so JS identifiers like `onChange = () => {}` do not false-positive.
    pattern = re.compile(r"""\son(?:click|focus|blur|change|input|load|error|submit|
                              keydown|keyup|keypress|mouseover|mouseout|dblclick)=["']""",
                         re.VERBOSE)
    offenders = []
    for f in list(ui_dir.glob("*.html")) + list(ui_dir.glob("*.js")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "inline event handlers are blocked by the CSP:\n" + "\n".join(offenders)


def test_ui_sources_contain_no_inline_style_attributes():
    """Same CSP, second trap: a `style="..."` ATTRIBUTE is inline style, so
    `default-src 'self'` blocks it too — and unlike a blocked handler this fails
    *visibly wrong* rather than silently. Measured live on /ui/: five ranked
    predictions carrying width:100%/0%/0%/0%/0% all rendered at the container's
    full 171 px, so a 0.004 label looked exactly like a 1.000 one. Widths belong
    in `data-width` + `el.style.width` (the CSSOM is not governed by style-src).
    """
    import re
    from pathlib import Path

    ui_dir = Path(__file__).parent.parent / "app" / "static" / "ui"
    # Attribute form only (` style="` / ` style='`), so a JS property assignment
    # like `el.style.width = ...` — the very fix this test asks for — is allowed.
    pattern = re.compile(r"""\sstyle=["']""")
    offenders = []
    for f in list(ui_dir.glob("*.html")) + list(ui_dir.glob("*.js")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "inline style attributes are blocked by the CSP:\n" + "\n".join(offenders)


def test_cors_wildcard_origin_disables_credentials(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path, APIV3_CORS_ALLOW_ORIGINS="*")
    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 200
    # With a wildcard origin, credentials must NOT be allowed.
    assert resp.headers.get("access-control-allow-credentials") is None


def test_config_endpoint_exposes_safe_fields_without_secrets(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)
    resp = client.get("/config", headers=RO)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auth_enabled"] is True
    assert "effective_n_jobs" in body
    # The safe config must never echo the configured API keys.
    assert "admin-key" not in resp.text and "ro-key" not in resp.text


def test_content_security_policy_covers_app_and_self_hosted_docs(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)
    csp = client.get("/health").headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    # /docs is now self-hosted (Swagger assets vendored, external init script, no
    # CDN), so it is same-origin and therefore also under CSP — no exemption.
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "cdn.jsdelivr.net" not in docs.text and "unpkg.com" not in docs.text
    assert "/swagger-static/swagger-ui-bundle.js" in docs.text
    assert "default-src 'self'" in docs.headers.get("Content-Security-Policy", "")
    # Every runtime dependency of the page resolves same-origin: the CSS, the
    # external init script, and the schema the init fetches.
    assert client.get("/swagger-static/swagger-ui.css").status_code == 200
    assert client.get("/swagger-static/swagger-init.js").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_health_and_config_response_models_preserve_exact_keys(monkeypatch, tmp_path):
    """The fixed-shape endpoints gained response_models; the serialized body must
    keep the exact same key set (a response_model that misses a field silently
    drops it, an extra field 500s on validation)."""
    client = _fresh_client(monkeypatch, tmp_path)

    health = client.get("/health")
    assert health.status_code == 200
    assert set(health.json()) == {"status", "version"}

    config = client.get("/config", headers=RO)
    assert config.status_code == 200, config.text
    assert set(config.json()) == {
        "n_jobs", "cpu_max_percent", "effective_n_jobs",
        "tfidf_max_word_features", "tfidf_max_char_features", "max_models_in_memory",
        "warmup_models", "auth_enabled", "rate_limit_enabled", "max_upload_mb",
    }
    # The response model must never leak a configured key.
    assert "admin-key" not in config.text and "ro-key" not in config.text


def test_default_limit_wires_rate_limit_default(monkeypatch, tmp_path):
    _fresh_client(monkeypatch, tmp_path,
                  APIV3_RATE_LIMIT_ENABLED="true", APIV3_RATE_LIMIT_DEFAULT="77/minute")
    from app.limiter import default_limit

    assert default_limit() == "77/minute"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Every test here mutates env + create_app; reset the settings AND registry
    singletons so the module-level app used by other test files re-resolves them
    from its own (process-permanent) environment afterward."""
    yield
    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()
