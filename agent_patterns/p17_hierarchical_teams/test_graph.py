import pytest

from agent_patterns.core import MockLLM
from agent_patterns.p17_hierarchical_teams.graph import (build_graph, mock_llm, parse_choice, EXAMPLE_INPUT, TEAMS,
                                                         FINISH, DONE, MAX_TEAM_TURNS, MAX_WORKER_TURNS, NO_ANSWER)


def test_ceo_runs_four_teams_and_each_lead_runs_its_own_members():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [team for team, _ in out["history"]] == ["design", "development", "marketing", "sales"]
    reports = dict(out["history"])
    assert "ux_designer:" in reports["design"] and "ui_designer:" in reports["design"]
    assert "account_exec:" in reports["sales"] and "sales_engineer:" not in reports["sales"]   # lead used one member
    assert out["answer"].startswith("[design]") and "[sales]" in out["answer"]                # the whole launch plan
    frontend_prompt = llm.prompts[10]
    assert "You are the frontend developer" in frontend_prompt and "swipe to snooze" in frontend_prompt  # sees design
    assert out["turns"] == 4 and llm.calls == 23   # CEO 5 + design 5 + development 5 + marketing 5 + sales 3


@pytest.mark.parametrize("reply,options,stop,expected", [
    ("Marketing.", TEAMS, FINISH, "marketing"),
    ("banana", TEAMS, FINISH, FINISH),
    ("**backend_dev**", TEAMS["development"]["workers"], DONE, "backend_dev"),
    ("DONE", TEAMS["design"]["workers"], DONE, DONE),
    ("", TEAMS["sales"]["workers"], DONE, DONE),
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
    script = dict(by_keyword={"You are the CEO": "development",                  # never says FINISH
                              "You lead the development team": "backend_dev",    # never says DONE
                              "You are the backend developer": "more code"})
    g, s, llm = run_both(script)
    assert g["turns"] == s["turns"] == MAX_TEAM_TURNS
    assert g["history"] == s["history"]
    assert all(report.count("backend_dev:") == MAX_WORKER_TURNS for _, report in g["history"])
    assert llm.calls == MAX_TEAM_TURNS * (1 + 2 * MAX_WORKER_TURNS)   # per team turn: CEO + lead/member pairs


def test_simple_matches_graph():
    from agent_patterns.p17_hierarchical_teams.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("answer", "history", "turns"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
