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
    from app.settings import get_settings

    get_settings.cache_clear()
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
    # The vendored asset resolves same-origin.
    assert client.get("/swagger-static/swagger-ui.css").status_code == 200


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
    """Every test here mutates env + create_app; reset the settings singleton so
    the module-level app used by other test files is not affected afterward."""
    yield
    from app.settings import get_settings

    get_settings.cache_clear()
