from agent_patterns.core import MockLLM
from agent_patterns.p10_orchestrator_workers.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_SECTIONS


def test_orchestrator_plans_sections_workers_fill_them():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["sections"] == ["Installation", "Usage", "License"]
    assert out["report"].startswith("## Installation\nRun `pip install todo-cli`.")
    assert out["report"].index("## Usage") < out["report"].index("## License")
    assert llm.calls == 4


def test_worker_count_is_capped():
    llm = MockLLM(responses=["\n".join(f"S{i}" for i in range(10))], by_keyword={"Section:": "text"})
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert len(out["sections"]) == MAX_SECTIONS == 5 and llm.calls == 6


def test_simple_matches_graph():
    from agent_patterns.p10_orchestrator_workers.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("sections", "report"):
        assert s[key] == g[key], key
    assert s_llm.calls == g_llm.calls
