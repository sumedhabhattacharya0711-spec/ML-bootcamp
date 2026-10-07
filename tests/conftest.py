import pytest


@pytest.fixture(autouse=True)
def no_llm_cache(monkeypatch):
    """Tests use fake LLM replies; the on-disk answer cache would mix them up."""
    monkeypatch.setenv("LLM_CACHE", "0")
