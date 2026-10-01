"""Cold loads, the model cache, and the threads behind every awaited call (audit 2026-09-30, R04).

Eight requests for a model nobody had asked for yet loaded it eight times in a row: each
waited for the disk lock and then read the bundle again, never looking at the cache the first
had filled. `/predict/multi` names up to five models against a default cache of two, so five
calls with the same three models made fifteen loads and not one hit. And every
`asyncio.to_thread` ran on the event loop's default executor -- min(32, CPUs + 4) threads, 8 on
four cores, not the 40 the comments assumed -- where an export and eight model reads kept a
prediction on a loaded model waiting 4.3 s.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import registry as registry_mod
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.schemas import MultiPredictRequest
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("models")
    settings = Settings(data_dir=FIXTURES, models_dir=directory, auth_enabled=False)
    profile = Profile("fast", "TF-IDF", True, True, [1.0])
    config = TrainingConfig(default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
                            test_size=0.2, min_text_length=5, drop_duplicates=True,
                            min_samples_per_label=2)
    run_training({"dataset_name": "tiny.csv", "model_name": "m",
                  "text_columns": ["properties.cclom:title"], "label_column": "properties.ccm:taxonid",
                  "csv_separator": ";", "label_separator": ",", "label_filter": None},
                 settings, config, profile, Registry(directory, 2),
                 on_progress=lambda **_: None, should_stop=lambda: False)
    return directory


def test_concurrent_requests_for_a_cold_model_load_it_once(models, monkeypatch):
    reads = []
    read_bundle = registry_mod._read_bundle
    monkeypatch.setattr(registry_mod, "_read_bundle",
                        lambda path: reads.append(path.name) or read_bundle(path))
    registry = Registry(models, 2)
    together = threading.Barrier(8)

    def ask(_):
        together.wait(5)
        return registry.get("m")

    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(ask, range(8)))

    assert reads == ["m"], f"{len(reads)} loads for one model"
    assert all(model is answers[0] for model in answers)


def test_the_cache_holds_every_model_one_multi_call_names():
    def accepted(count: int) -> bool:
        try:
            MultiPredictRequest(texts=["x"], model_names=[f"m{i}" for i in range(count)])
        except ValidationError:
            return False
        return True

    most = max(count for count in range(1, 20) if accepted(count))

    assert Settings(max_models_in_memory=2).effective_max_models_in_memory() >= most


def test_awaited_work_gets_as_many_threads_as_the_routes_pool(monkeypatch, tmp_path):
    """33 calls at once: one more than the default executor ever has on any machine."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    for key, value in {"APIV3_API_KEY_ADMIN": "admin-key", "APIV3_API_KEY_READONLY": "ro-key",
                       "APIV3_DATA_DIR": str(tmp_path / "data"),
                       "APIV3_MODELS_DIR": str(tmp_path / "models")}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    get_registry.cache_clear()

    async def at_once() -> int:
        together = threading.Barrier(33)
        await asyncio.gather(*(asyncio.to_thread(together.wait, 5) for _ in range(33)))
        return 33

    try:
        with TestClient(create_app()) as client:
            assert client.portal.call(at_once) == 33
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()
