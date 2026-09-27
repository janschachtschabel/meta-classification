"""Auth-disabled is local use only — and the code now says so too (audit 2026-09-27, S-1).

The docs have always called `APIV3_AUTH_ENABLED=false` "local use only", and data-prep enforces
the same rule; api_v3 did not. With auth off, `require_role` answered "admin" for every caller,
so one environment variable plus any non-loopback exposure — a `-p 8000:8000`, the chart's
ingress — was an unauthenticated admin API: train, delete, import.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import get_registry
from app.settings import get_settings


@pytest.fixture
def keyless(monkeypatch, tmp_path):
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "false")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    get_registry.cache_clear()
    yield
    get_settings.cache_clear()
    get_registry.cache_clear()


@pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.1.1", "::1"])
def test_keyless_mode_serves_a_caller_on_this_machine(keyless, peer):
    with TestClient(create_app(), client=(peer, 50000)) as client:
        assert client.get("/models").status_code == 200


@pytest.mark.parametrize("peer", ["203.0.113.7", "10.42.0.9", "172.17.0.1"])
def test_keyless_mode_refuses_a_caller_from_the_network(keyless, peer):
    """172.17.0.1 is Docker's bridge gateway: a request from the host into a container arrives
    from there, not from loopback. So keyless mode does not work inside a container — as in
    data-prep, by design. A key is the way to run it there, and the refusal says which."""
    with TestClient(create_app(), client=(peer, 50000)) as client:
        answer = client.get("/models")

    assert answer.status_code == 403
    assert "APIV3_AUTH_ENABLED" in answer.json()["detail"]


def test_an_admin_route_is_refused_from_the_network_too(keyless):
    with TestClient(create_app(), client=("203.0.113.7", 50000)) as client:
        assert client.delete("/models/anything").status_code == 403


def test_the_public_probes_stay_public_whoever_asks(keyless):
    """A kubelet probing /health or /ready sends no key and is not on loopback."""
    with TestClient(create_app(), client=("10.42.0.9", 50000)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
