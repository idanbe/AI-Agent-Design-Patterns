import pytest

from agent_patterns.core import MockLLM
from agent_patterns.p17_hierarchical_teams.graph import (build_graph, mock_llm, parse_choice, EXAMPLE_INPUT, TEAMS,
                                                         FINISH, DONE, MAX_TEAM_TURNS, MAX_WORKER_TURNS, NO_ANSWER)

FINAL_LINE = "HydroSmart: the smart bottle that keeps you hydrated. Launch price $39."


def test_ceo_runs_research_then_marketing_and_each_lead_runs_its_own_team():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [team for team, _ in out["history"]] == ["research", "marketing"]
    research_report = out["history"][0][1]
    assert "market_researcher:" in research_report and "data_analyst:" in research_report
    assert out["answer"] == FINAL_LINE
    copywriter_prompt = llm.prompts[8]
    assert "You are the copywriter" in copywriter_prompt and "gym-goers" in copywriter_prompt  # sees research's report
    assert out["turns"] == 2 and llm.calls == 13       # CEO 3 + research lead 3 + 2 workers + marketing lead 3 + 2 workers


@pytest.mark.parametrize("reply,options,stop,expected", [
    ("Marketing.", TEAMS, FINISH, "marketing"),
    ("banana", TEAMS, FINISH, FINISH),
    ("**data_analyst**", TEAMS["research"]["workers"], DONE, "data_analyst"),
    ("DONE", TEAMS["research"]["workers"], DONE, DONE),
    ("", TEAMS["marketing"]["workers"], DONE, DONE),
])
def test_parse_choice_normalizes_and_stops_on_unknown(reply, options, stop, expected):
    assert parse_choice(reply, options, stop) == expected


def run_both(script):
    from agent_patterns.p17_hierarchical_teams.simple import run
    g_llm, s_llm = MockLLM(**script), MockLLM(**script)
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert g_llm.calls == s_llm.calls
    return g, s, g_llm


def test_garbage_ceo_reply_stops_at_once_in_both_versions():
    g, s, llm = run_both(dict(responses=["not sure"]))
    assert g["answer"] == s["answer"] == NO_ANSWER and g["turns"] == s["turns"] == 0 and llm.calls == 1


def test_both_levels_are_capped_in_both_versions():
    script = dict(by_keyword={"You are the CEO": "research",                    # never says FINISH
                              "You lead the research team": "data_analyst",    # never says DONE
                              "You are the data analyst": "more numbers"})
    g, s, llm = run_both(script)
    assert g["turns"] == s["turns"] == MAX_TEAM_TURNS
    assert g["history"] == s["history"]
    assert all(report.count("data_analyst:") == MAX_WORKER_TURNS for _, report in g["history"])
    assert llm.calls == MAX_TEAM_TURNS * (1 + 2 * MAX_WORKER_TURNS)   # per team turn: CEO + lead/worker pairs


def test_simple_matches_graph():
    from agent_patterns.p17_hierarchical_teams.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("answer", "history", "turns"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
