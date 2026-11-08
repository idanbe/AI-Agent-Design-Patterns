from agent_patterns.core import MockLLM
from agent_patterns.p14_tree_of_thoughts.graph import build_graph, mock_llm, apply_step, EXAMPLE_INPUT


def test_tot_beam_search_finds_24():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["solution"] == ["10 - 4 = 6", "13 - 9 = 4", "6 * 4 = 24"]
    assert llm.calls == 10


def test_tot_gives_up_when_every_branch_is_impossible():
    llm = MockLLM(responses=["impossible"], by_keyword={"Numbers: 4 9 10 13\n": "10 - 4 = 6\n13 - 9 = 4"})
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out.get("solution") is None and out["frontier"] == [] and llm.calls == 3


def test_apply_step_rechecks_model_arithmetic():
    assert apply_step([4, 9, 10, 13], "10 - 4 = 6") == {"nums": [9, 13, 6.0], "step": "10 - 4 = 6"}
    assert apply_step([4, 9, 10, 13], "10 - 4 = 7") is None      # wrong math
    assert apply_step([4, 9, 10, 13], "5 + 5 = 10") is None      # numbers not available
    assert apply_step([4, 9, 10, 13], "no step here") is None


def test_simple_matches_graph():
    from agent_patterns.p14_tree_of_thoughts.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    assert s["solution"] == g["solution"]
    assert s_llm.calls == g_llm.calls


import pytest  # noqa: E402


@pytest.mark.parametrize("line", ["1. 10 - 4 = 6", "2) 10 - 4 = 6", "- 10 - 4 = 6", "* 10 - 4 = 6"])
def test_apply_step_accepts_list_markers(line):
    assert apply_step([4, 9, 10, 13], line) == {"nums": [9, 13, 6.0], "step": "10 - 4 = 6"}
