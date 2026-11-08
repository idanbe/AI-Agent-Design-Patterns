from agent_patterns.core import MockLLM
from agent_patterns.p04_reflexion.graph import build_graph, mock_llm, check_slogan, EXAMPLE_INPUT, MAX_TRIALS


def test_reflexion_learns_from_external_feedback():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["passed"] and out["trials"] == 2
    assert out["attempt"] == "Refill daily, protect the planet."
    assert len(out["reflections"]) == 1
    assert out["reflections"][0] in llm.prompts[2]     # 2nd attempt saw the lesson
    assert llm.calls == 3


def test_reflexion_stops_after_max_trials():
    llm = MockLLM(responses=["Buy now!"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert not out["passed"] and out["trials"] == MAX_TRIALS == 3
    assert llm.calls == 5          # 3 attempts + 2 reflections


def test_check_slogan():
    assert check_slogan("Refill daily, protect the planet.") == []
    assert len(check_slogan("Stay hydrated and save the planet every single day!")) == 2


def test_simple_matches_graph():
    from agent_patterns.p04_reflexion.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("attempt", "passed", "trials", "reflections"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
