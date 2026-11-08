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


import pytest  # noqa: E402


@pytest.mark.parametrize("reply", ["Follow up:", "", "So the final answer is:"])
def test_empty_or_broken_reply_ends_with_an_answer_in_both_versions(reply):
    from agent_patterns.p12_self_ask.simple import run
    g_llm, s_llm = MockLLM(responses=[reply]), MockLLM(responses=[reply])
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert g["answer"] and g["answer"] == s["answer"]
    assert g_llm.calls == s_llm.calls == 1


def test_empty_reply_after_a_follow_up_still_stops():
    from agent_patterns.p12_self_ask.simple import run
    script = dict(responses=["Follow up: q?"], by_keyword={"\nFollow up: q?": ""})
    g_llm, s_llm = MockLLM(**script), MockLLM(**script)
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT, {"recursion_limit": 25})
    s = run(EXAMPLE_INPUT, s_llm)
    assert g["answer"] and g["answer"] == s["answer"]
    assert g_llm.calls == s_llm.calls == 2
