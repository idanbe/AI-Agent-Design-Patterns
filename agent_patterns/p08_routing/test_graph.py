import pytest
from agent_patterns.core import MockLLM
from agent_patterns.p08_routing.graph import build_graph, mock_llm, EXAMPLE_INPUT


def test_example_routes_to_billing():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["route"] == "billing" and "refunded" in out["answer"] and llm.calls == 2


@pytest.mark.parametrize("label,route", [("billing", "billing"), ("Technical.", "technical"), ("banana", "general")])
def test_router_normalizes_label_and_falls_back(label, route):
    llm = MockLLM(responses=[label, f"{route} answer"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["route"] == route and out["answer"] == f"{route} answer"


def test_simple_matches_graph():
    from agent_patterns.p08_routing.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("route", "answer"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
