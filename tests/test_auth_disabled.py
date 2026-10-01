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



def test_a_proxy_cannot_vouch_for_loopback(keyless):
    """S04 (audit 2026-09-30): uvicorn rewrites the peer from X-Forwarded-For for every peer
    FORWARDED_ALLOW_IPS trusts, and the chart suggested the ingress controller's pod CIDR
    there -- so any pod in it sending `X-Forwarded-For: 127.0.0.1` was keyless admin. The
    rewritten peer is all the app gets to see, so the chain is tested as deployed."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    behind_a_proxy = ProxyHeadersMiddleware(create_app(), trusted_hosts="10.0.0.0/8")
    with TestClient(behind_a_proxy, client=("10.42.0.7", 50000)) as client:
        answer = client.delete("/models/anything", headers={"X-Forwarded-For": "127.0.0.1"})

    assert answer.status_code == 403, answer.text


@pytest.mark.parametrize("header", ["X-Forwarded-For", "Forwarded", "X-Real-IP"])
def test_a_request_relayed_by_a_proxy_on_this_machine_is_not_local(keyless, header):
    """A reverse proxy on this machine connects from loopback, whoever it relays: the
    header it adds says the caller is someone else, and keyless mode serves only this
    machine. A key is the way to put a proxy in front."""
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
        answer = client.get("/models", headers={header: "for=203.0.113.7" if header == "Forwarded" else "203.0.113.7"})

    assert answer.status_code == 403
