from agent_patterns.core import MockLLM
from agent_patterns.p05_self_refine.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_ITERATIONS


def test_self_refine_iterates_until_no_issues():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["draft"] == "fix(profile): handle users without an email address"
    assert out["history"] == ["fixed bug"]
    assert out["iterations"] == 2 and llm.calls == 4


def test_self_refine_stops_at_max_iterations():
    llm = MockLLM(responses=["meh"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["iterations"] == MAX_ITERATIONS == 3
    assert llm.calls == 6          # generate + 3 feedback + 2 refine


def test_simple_matches_graph():
    from agent_patterns.p05_self_refine.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("draft", "history", "iterations"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
