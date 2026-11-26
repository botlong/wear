"""Checks, against the installed LiteLLM, that the decoding kwargs run_batch.py
resolves for routed models survive into the request (run in the tau2 env)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as rb  # noqa: E402


def _sent(model, monkeypatch):
    """Optional params LiteLLM would send for a routed model with run_batch's kwargs."""
    monkeypatch.setitem(rb.ROUTES, model, "RELAY_X")
    monkeypatch.setenv("RELAY_X_API_KEY", "k")
    monkeypatch.setenv("RELAY_X_BASE_URL", "https://example.test")
    extra = rb._endpoint_for(model)[4]
    kw = dict(json.loads(extra.get("TAU2_LLM_EXTRA_KWARGS", "{}")))
    kw.update(json.loads(extra.get("TAU2_LLM_PER_MODEL_KWARGS", "{}")).get(model, {}))
    kw.pop("timeout", None)
    litellm.drop_params = True  # as tau2's llm_utils
    from litellm.utils import get_optional_params
    return get_optional_params(model=model.split("/", 1)[1], custom_llm_provider="openai",
                               seed=300, **kw)


@pytest.mark.parametrize("model", ["openai/gemini-3.5-flash", "openai/gemini-3.1-pro-preview"])
def test_gemini_reasoning_effort_reaches_request(model, monkeypatch):
    p = _sent(model, monkeypatch)
    assert p.get("reasoning_effort") == "low" and p.get("seed") == 300


def test_deepseek_keeps_seed_and_cap(monkeypatch):
    p = _sent("openai/deepseek-v4-pro", monkeypatch)
    assert p.get("seed") == 300 and p.get("max_tokens") == 8192
