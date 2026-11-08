"""Real-model smoke test: every pattern finishes on real OpenRouter output. Skipped without a key.

Run: uv run --env-file .env pytest tests/test_openrouter_smoke.py -q
"""
import importlib
import os

import pytest

from agent_patterns.core import PATTERNS

pytestmark = pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")


@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_completes_with_real_model(name, monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    mod = importlib.import_module(f"agent_patterns.{name}.graph")
    out = mod.build_graph().invoke(mod.EXAMPLE_INPUT, {"recursion_limit": 40})
    assert out    # finishes on real-model text (format fallbacks hold)
