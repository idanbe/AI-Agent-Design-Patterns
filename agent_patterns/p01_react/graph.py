"""01 · ReAct: the LLM alternates thinking and tool calls, reading each tool result before its next move."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent_patterns.core import MockLLM, get_llm, run_demo

SYSTEM = ("You are a careful assistant. Think step by step. Call a tool whenever you need a fact "
          "or a calculation; when you have everything, answer in one sentence.")

FACTS = {"population of france": "68 million"}


@tool
def search(query: str) -> str:
    """Look up a fact, e.g. 'population of France'."""
    return next((v for k, v in FACTS.items() if k in query.lower()), "no result")


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


TOOLS = [search, multiply]
MAX_TURNS = 10   # LLM turns; LangGraph's own recursion_limit defaults to 10007 in 1.2.x - too high to be the guard
STOP_ANSWER = "Stopped: turn limit reached."


class State(MessagesState):
    turns: int


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        AIMessage("", tool_calls=[{"name": "search", "args": {"query": "population of France"}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "multiply", "args": {"a": 68, "b": 2}, "id": "c2"}]),
        "Twice France's population is about 136 million.",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm
    model = llm.bind_tools(TOOLS)

    def agent(state: State):
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:                      # no tool calls in this message -> tools_condition ends
            return {"messages": [AIMessage(STOP_ANSWER)]}
        return {"messages": [model.invoke([SystemMessage(SYSTEM)] + state["messages"])], "turns": turns + 1}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, ["tools", "__end__"])
    g.add_edge("tools", "agent")
    return g.compile()


EXAMPLE_INPUT = {"messages": [HumanMessage("What is twice the population of France?")]}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
