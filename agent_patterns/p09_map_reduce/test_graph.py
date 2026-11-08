from agent_patterns.core import MockLLM
from agent_patterns.p09_map_reduce.graph import build_graph, mock_llm, EXAMPLE_INPUT


def test_map_in_parallel_then_reduce():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert sorted(out["summaries"]) == ["Battery life is excellent", "Good value for the price", "Screen is too dim"]
    assert out["summary"].startswith("Great battery") and llm.calls == 4


def test_fan_out_scales_with_input():
    llm = MockLLM(by_keyword={"Combine": "ok", "x": "s"})
    out = build_graph(llm).invoke({"reviews": ["x"] * 5, "summaries": []})
    assert len(out["summaries"]) == 5 and llm.calls == 6


def test_no_reviews_skips_llm():
    llm = MockLLM()
    out = build_graph(llm).invoke({"reviews": [], "summaries": []})
    assert out["summary"] == "No reviews." and llm.calls == 0


def test_simple_matches_graph():
    from agent_patterns.p09_map_reduce.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert sorted(s["summaries"]) == sorted(g["summaries"])
    assert s["summary"] == g["summary"]
    assert s_llm.calls == g_llm.calls
