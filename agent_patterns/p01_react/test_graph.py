from langchain_core.messages import ToolMessage
from agent_patterns.p01_react.graph import build_graph, mock_llm, search, EXAMPLE_INPUT


def test_react_loops_thought_action_observation():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [m.content for m in out["messages"] if isinstance(m, ToolMessage)] == ["68 million", "136.0"]
    assert "136" in out["messages"][-1].content
    assert llm.calls == 3


def test_search_miss_returns_no_result():
    assert search.invoke({"query": "population of Mars"}) == "no result"


def test_simple_matches_graph():
    from agent_patterns.p01_react.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert s["answer"] == g["messages"][-1].content
    assert s["observations"] == [m.content for m in g["messages"] if isinstance(m, ToolMessage)]
    assert s_llm.calls == g_llm.calls
