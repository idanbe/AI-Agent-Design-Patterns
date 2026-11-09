import pytest

from agent_patterns.core import MockLLM
from agent_patterns.p15_lats.graph import (build_graph, mock_llm, parse_actions, uct, backprop, Node,
                                           EXAMPLE_INPUT, MAX_ITERATIONS)


def test_lats_backtracks_after_a_failed_path_and_solves():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["solution"] == ["+3", "+1", "*2"]              # 1 -> 4 -> 5 -> 10
    assert len(out["reflections"]) == 1                        # 1 -> 4 -> 7 -> 14 overshot first
    assert out["reflections"][0] in llm.prompts[-1]            # the lesson reached the next proposal
    assert out["iterations"] == 4 and llm.calls == 9


def test_parse_actions_keeps_valid_samples_only():
    assert parse_actions("1. +3\n- *2\n+3\nmultiply by 5\n+ 1") == ["+3", "*2", "+3", "+1"]


def test_uct_explores_less_visited_children():
    parent = Node(1)
    a, b = Node(4, "+3", parent), Node(2, "*2", parent)
    parent.visits, a.visits, a.value, b.visits, b.value = 3, 3, 0.5, 1, 0.5
    assert uct(b, parent.visits) > uct(a, parent.visits)       # same value, fewer visits -> explore


def test_backprop_updates_running_mean_up_to_root():
    root = Node(1)
    child = Node(4, "+3", root)
    child.visits, child.value = 1, 0.6
    backprop(child, 0.0)
    assert child.visits == 2 and child.value == pytest.approx(0.3)
    assert root.visits == 1 and root.value == 0.0


DEAD_ENDS = [
    dict(responses=["no idea"]),                                                    # no usable action
    dict(responses=["+1"], by_keyword={"This path failed": "lesson", "Rate:": "5"}),  # one chain, dies at depth 3
]


@pytest.mark.parametrize("script", DEAD_ENDS)
def test_dead_tree_stops_in_both_versions(script):
    from agent_patterns.p15_lats.simple import run
    g_llm, s_llm = MockLLM(**script), MockLLM(**script)
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert g.get("solution") is None and s["solution"] is None
    assert g["iterations"] == s["iterations"] and g_llm.calls == s_llm.calls


def test_search_is_capped_in_both_versions():
    from agent_patterns.p15_lats.simple import run
    script = dict(responses=["+1\n+3\n*2"], by_keyword={"This path failed": "lesson", "Rate:": "5"})
    unreachable = {**EXAMPLE_INPUT, "start": 0}                # 10 can't be reached from 0 in 3 steps
    g_llm, s_llm = MockLLM(**script), MockLLM(**script)
    g = build_graph(g_llm).invoke(unreachable)
    s = run(unreachable, s_llm)
    assert g.get("solution") is None and s["solution"] is None
    assert g["iterations"] == s["iterations"] == MAX_ITERATIONS
    assert g_llm.calls == s_llm.calls


def test_simple_matches_graph():
    from agent_patterns.p15_lats.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("solution", "reflections", "iterations"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
