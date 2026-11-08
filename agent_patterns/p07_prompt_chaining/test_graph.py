from agent_patterns.core import MockLLM
from agent_patterns.p07_prompt_chaining.graph import build_graph, mock_llm, EXAMPLE_INPUT


def test_chain_runs_fixed_steps_in_order():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["outline"].count("- ") == 3
    assert "recycled steel" in out["draft"]
    assert "בקבוק" in out["translation"]
    assert llm.calls == 3


def test_gate_stops_chain_on_bad_outline():
    llm = MockLLM(responses=["- only one point"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert "outline" in out["error"] and "draft" not in out and llm.calls == 1


def test_simple_matches_graph():
    from agent_patterns.p07_prompt_chaining.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("outline", "draft", "translation"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
