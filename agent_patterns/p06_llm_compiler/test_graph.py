from agent_patterns.core import MockLLM
from agent_patterns.p06_llm_compiler.graph import build_graph, mock_llm, parse_tasks, MOCK_PLAN, EXAMPLE_INPUT


def test_independent_tasks_run_in_parallel_waves():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["waves"] == [[1, 2], [3]]
    assert out["results"] == {1: "125", 2: "84", 3: "41"}
    assert "41" in out["answer"] and llm.calls == 2


def test_empty_plan_goes_straight_to_joiner():
    llm = MockLLM(responses=["no tools needed", "answer"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["waves"] == [] and out["answer"] == "answer"


def test_parse_tasks_extracts_dependencies():
    assert parse_tasks(MOCK_PLAN)[3] == {"tool": "subtract", "arg": "$1, $2", "deps": {1, 2}}


def test_simple_matches_graph():
    from agent_patterns.p06_llm_compiler.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("waves", "results", "answer"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
