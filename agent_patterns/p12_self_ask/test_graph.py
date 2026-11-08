from agent_patterns.core import MockLLM
from agent_patterns.p12_self_ask.graph import build_graph, mock_llm, search, EXAMPLE_INPUT, MAX_HOPS


def test_self_ask_decomposes_into_follow_ups():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["answer"] == "Benjamin Harrison"
    assert out["scratchpad"].count("Intermediate answer:") == 2
    assert "Intermediate answer: 1889" in out["scratchpad"]
    assert llm.calls == 3


def test_endless_follow_ups_are_capped():
    llm = MockLLM(responses=["Follow up: again?"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["answer"].startswith("unknown") and llm.calls == MAX_HOPS == 4


def test_search_miss_is_unknown():
    assert search("What is the GDP of Atlantis?") == "unknown"


def test_simple_matches_graph():
    from agent_patterns.p12_self_ask.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("answer", "scratchpad"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
