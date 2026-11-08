from agent_patterns.core import MockLLM
from agent_patterns.p02_rewoo.graph import build_graph, mock_llm, EXAMPLE_INPUT


def test_rewoo_plans_once_runs_tools_then_solves():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["steps"] == [("#E1", "Search", "2018 FIFA World Cup winner"), ("#E2", "Search", "capital of #E1")]
    assert out["results"] == {"#E1": "France", "#E2": "Paris"}
    assert "Paris" in out["result"]
    assert llm.calls == 2          # planner + solver only


def test_unparseable_plan_goes_straight_to_solver():
    llm = MockLLM(responses=["I will just answer.", "Paris"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["results"] == {} and out["result"] == "Paris" and llm.calls == 2


def test_simple_matches_graph():
    from agent_patterns.p02_rewoo.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert s["results"] == g["results"]
    assert s["result"] == g["result"]
    assert s_llm.calls == g_llm.calls
