import pytest

from agent_patterns.core import MockLLM
from agent_patterns.p16_supervisor.graph import (build_graph, mock_llm, parse_next, EXAMPLE_INPUT,
                                                 FINISH, MAX_TURNS, NO_ANSWER)


def test_supervisor_picks_one_worker_at_a_time_until_finish():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [who for who, _ in out["history"]] == ["researcher", "analyst", "writer"]
    assert out["answer"] == "Summary: Q3 revenue grew 50% QoQ, from $80k to $120k."
    assert "You are the analyst" in llm.prompts[3] and "Q3 revenue $120k" in llm.prompts[3]  # sees shared work
    assert out["turns"] == 3 and llm.calls == 7                                           # 4 decisions + 3 workers


@pytest.mark.parametrize("reply,expected", [("analyst", "analyst"), ("Writer.", "writer"), ("**researcher**", "researcher"),
                                            ("FINISH", FINISH), ("banana", FINISH), ("", FINISH)])
def test_parse_next_normalizes_and_stops_on_unknown(reply, expected):
    assert parse_next(reply) == expected


SCRIPTS = {
    "garbage_first_reply": (dict(responses=["I am not sure"]), 0, NO_ANSWER),          # stops at once, no answer
    "never_finishes": (dict(responses=["researcher", "more facts"]), MAX_TURNS, "more facts"),  # capped
}


@pytest.mark.parametrize("name", SCRIPTS)
def test_fallbacks_and_cap_agree_in_both_versions(name):
    from agent_patterns.p16_supervisor.simple import run
    script, turns, answer = SCRIPTS[name]
    g_llm, s_llm = MockLLM(**script), MockLLM(**script)
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert g["turns"] == s["turns"] == turns
    assert g["answer"] == s["answer"] == answer
    assert g_llm.calls == s_llm.calls


def test_simple_matches_graph():
    from agent_patterns.p16_supervisor.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("answer", "history", "turns"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
