"""08 · Routing: classify the input, then send it to exactly one specialized handler."""
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

ROUTES = ("billing", "technical", "general")

CLASSIFY = """Classify this support ticket as exactly one word: billing, technical or general.

Ticket: {ticket}"""

HANDLER_PROMPTS = {
    "billing": "You are a billing specialist. Explain charges and refunds briefly and kindly.\n\nTicket: {ticket}",
    "technical": "You are a support engineer. Give clear troubleshooting steps.\n\nTicket: {ticket}",
    "general": "You are a friendly support agent. Answer helpfully and briefly.\n\nTicket: {ticket}",
}


class State(TypedDict):
    ticket: str
    route: str
    answer: str


def parse_route(reply: str) -> str:
    """Normalize the label; anything unexpected goes to the general handler."""
    route = reply.strip().lower().strip(".")
    return route if route in ROUTES else "general"


def mock_llm() -> MockLLM:
    return MockLLM(responses=["billing", "You were charged twice; we've refunded the duplicate payment."])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def classify(state: State):
        return {"route": parse_route(str(llm.invoke(CLASSIFY.format(ticket=state["ticket"])).content))}

    def make_handler(route: str):
        def handle(state: State):
            return {"answer": str(llm.invoke(HANDLER_PROMPTS[route].format(ticket=state["ticket"])).content)}
        return handle

    g = StateGraph(State)
    g.add_node("classify", classify)
    for route in ROUTES:
        g.add_node(route, make_handler(route))
        g.add_edge(route, END)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", lambda s: s["route"], {r: r for r in ROUTES})
    return g.compile()


EXAMPLE_INPUT = {"ticket": "I was charged twice for my subscription this month."}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
