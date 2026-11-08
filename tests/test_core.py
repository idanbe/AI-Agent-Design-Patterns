import pytest
from typing import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_patterns.core import (MockLLM, get_llm, run_demo, PATTERNS,
                                 OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL)


def test_mock_replays_in_order_and_cycles():
    llm = MockLLM(responses=["a", "b"])
    assert [llm.invoke("x").content for _ in range(3)] == ["a", "b", "a"]
    assert llm.calls == 3 and llm.prompts == ["x", "x", "x"]


def test_mock_keyword_wins_and_does_not_advance_sequence():
    llm = MockLLM(responses=["seq"], by_keyword={"weather": "sunny"})
    assert llm.invoke("what is the weather?").content == "sunny"
    assert llm.invoke("other").content == "seq"


def test_mock_returns_tool_calls_and_bind_tools_is_noop():
    call = AIMessage("", tool_calls=[{"name": "add", "args": {"a": 1}, "id": "1"}])
    llm = MockLLM(responses=[call])
    assert llm.bind_tools([]) is llm
    assert llm.invoke("x").tool_calls[0]["name"] == "add"


def test_mock_without_script_raises_clear_error():
    with pytest.raises(ValueError, match="no scripted reply"):
        MockLLM().invoke("hello")


def test_get_llm_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("USE_OPENROUTER", raising=False)
    assert isinstance(get_llm(lambda: MockLLM(responses=["x"])), MockLLM)


def test_get_llm_uses_openrouter_when_enabled(monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    llm = get_llm(MockLLM)
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.model_name == DEFAULT_OPENROUTER_MODEL


def test_get_llm_openrouter_without_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_llm(MockLLM)


def test_run_demo_prints_nodes_and_returns_final_state(capsys):
    class S(TypedDict):
        n: int
    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc"); g.add_edge("inc", END)
    assert run_demo(g.compile(), {"n": 1}) == {"n": 2}
    assert "▶ inc" in capsys.readouterr().out


def test_patterns_discovered_and_numbered_consecutively():
    assert [int(n[1:3]) for n in PATTERNS] == list(range(1, len(PATTERNS) + 1))


def test_run_demo_prints_messages_readably(capsys):
    from langgraph.graph import MessagesState
    call = AIMessage("", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}])
    g = StateGraph(MessagesState)
    g.add_node("agent", lambda s: {"messages": [call, AIMessage("done")]})
    g.add_edge(START, "agent"); g.add_edge("agent", END)
    run_demo(g.compile(), {"messages": []})
    out = capsys.readouterr().out
    assert "ai → search({'query': 'x'})" in out and "ai: done" in out
    assert "additional_kwargs" not in out


def test_mock_replays_fresh_message_objects():
    # reusing one AIMessage object makes add_messages *replace* instead of append on replay
    call = AIMessage("", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "z3"}])
    llm = MockLLM(responses=[call])
    first, second = llm.invoke("x"), llm.invoke("x")
    assert first is not second and first.id != second.id
