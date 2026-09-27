"""How many bulk classifications may hold a threadpool worker at once (audit PERF-2).

`POST /predict/csv` streams a **sync** generator, which Starlette iterates on one anyio worker
for the whole classification — minutes for a 200 MB CSV — out of the same 40-worker pool every
`def` route and every `asyncio.to_thread` uses. Nothing bounded it, and the rate limit is per
client address with a readonly key.

The slot arithmetic is tested directly; the route's use of it is tested through the API, because
the part worth pinning is *when* the slot is taken (after the upload has been validated) and
*that* it comes back on every exit path, including a client that walks away.
"""

import threading

import pytest

from app.concurrency import Slots

# --- the counter -----------------------------------------------------------------------------


def test_slots_refuse_rather_than_wait():
    slots = Slots(2)

    assert slots.try_acquire() is True
    assert slots.try_acquire() is True
    assert slots.try_acquire() is False, "a third caller was let through"
    assert slots.in_use == 2


def test_a_released_slot_is_available_again():
    slots = Slots(1)
    assert slots.try_acquire() is True
    assert slots.try_acquire() is False

    slots.release()

    assert slots.try_acquire() is True


def test_releasing_more_often_than_acquired_cannot_create_capacity():
    """The release sits in a generator's `finally`, which a client disconnect can reach on a
    path the background task also runs. A double release must not hand out a slot that does
    not exist — which is why this is not a `threading.Semaphore` (it raises there) nor a bare
    decrement (it would go negative and uncap the route)."""
    slots = Slots(1)
    slots.try_acquire()

    slots.release()
    slots.release()
    slots.release()

    assert slots.in_use == 0
    assert slots.try_acquire() is True
    assert slots.try_acquire() is False, "the extra releases created capacity"


def test_a_capacity_below_one_still_allows_one():
    """A misconfigured 0 would otherwise turn the endpoint off entirely — a config mistake
    should not look like a broken route."""
    assert Slots(0).try_acquire() is True
    assert Slots(-5).try_acquire() is True


def test_the_counter_holds_under_concurrent_callers():
    """It is acquired from the event loop and released from a worker thread, so the count is
    guarded rather than assumed atomic."""
    slots = Slots(10)
    granted: list[bool] = []
    barrier = threading.Barrier(20)

    def race():
        barrier.wait(timeout=5)
        granted.append(slots.try_acquire())

    threads = [threading.Thread(target=race) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sum(granted) == 10, f"{sum(granted)} of 20 callers got one of 10 slots"
    assert slots.in_use == 10


# --- the route -------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import concurrency
    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_MAX_CONCURRENT_CSV", "1")
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    get_settings.cache_clear()
    get_registry.cache_clear()
    concurrency.reset_csv_slots()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
    get_registry.cache_clear()
    concurrency.reset_csv_slots()


def test_the_limit_comes_from_the_setting(client):
    from app.concurrency import csv_slots

    assert csv_slots().capacity == 1


def test_a_request_that_never_reaches_the_stream_does_not_hold_a_slot(client):
    """Claimed after the upload and the column check: a 400 for a missing column must not
    cost a slot, or a client sending bad requests could close the endpoint for everyone."""
    from app.concurrency import csv_slots

    files = {"file": ("x.csv", b"nope;other\na;b\n", "text/csv")}
    response = client.post(
        "/predict/csv", files=files,
        data={"model_name": "missing_model", "text_columns": "nope"},
        headers={"X-API-Key": "ro-key"},
    )

    assert response.status_code in (400, 404, 422), response.text
    assert csv_slots().in_use == 0, "a refused request kept a slot"
