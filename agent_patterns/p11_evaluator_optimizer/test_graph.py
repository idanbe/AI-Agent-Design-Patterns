from agent_patterns.core import MockLLM
from agent_patterns.p11_evaluator_optimizer.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_ATTEMPTS


def test_loops_until_judge_passes():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["grade"] == "pass" and out["attempts"] == 2
    assert out["draft"] == "איזה כיף, ההזמנה שלך כבר בדרך אליך!"
    assert "cold and formal" in llm.prompts[2]       # generator saw the judge's feedback
    assert llm.calls == 4


def test_unparseable_grade_is_fail_and_loop_is_capped():
    llm = MockLLM(responses=["draft", "looks fine I guess"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["grade"] == "fail" and out["attempts"] == MAX_ATTEMPTS == 3 and llm.calls == 6


def test_simple_matches_graph():
    from agent_patterns.p11_evaluator_optimizer.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("draft", "grade", "attempts"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
