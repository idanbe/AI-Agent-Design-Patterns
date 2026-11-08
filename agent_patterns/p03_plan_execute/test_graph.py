from agent_patterns.core import MockLLM
from agent_patterns.p03_plan_execute.graph import build_graph, mock_llm, parse_plan, EXAMPLE_INPUT, MAX_STEPS


def test_plan_execute_replans_after_each_step():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [s for s, _ in out["past_steps"]] == ["Look up the hotel price per night in Paris",
                                                 "Add a daily food budget of 60 EUR"]
    assert out["response"].startswith("Budget about 210 EUR")
    assert llm.calls == 5


def test_replanner_that_never_finishes_is_capped():
    llm = MockLLM(responses=["1. step"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert len(out["past_steps"]) == MAX_STEPS == 6
    assert out["response"].startswith("Stopped")
    assert llm.calls == 12         # planner + 6 executor + 5 replanner


def test_parse_plan_tolerates_noise():
    assert parse_plan("1. a\n2) b\nnoise") == ["a", "b"]


def test_simple_matches_graph():
    from agent_patterns.p03_plan_execute.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert s["response"] == g["response"]
    assert s["past_steps"] == g["past_steps"]
    assert s_llm.calls == g_llm.calls


import pytest  # noqa: E402


@pytest.mark.parametrize("last", ["FINAL:", ""])
def test_empty_replanner_reply_ends_the_run_in_both_versions(last):
    from agent_patterns.p03_plan_execute.simple import run
    script = ["1. a", "r", last]
    g_llm, s_llm = MockLLM(responses=script), MockLLM(responses=script)
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert g["response"] and g["response"] == s["response"]
    assert g["past_steps"] == s["past_steps"] == [("a", "r")]
    assert g_llm.calls == s_llm.calls == 3
