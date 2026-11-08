"""09 · Parallelization (map-reduce): run the same LLM step on every input in parallel, then combine."""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_patterns.core import MockLLM, get_llm, run_demo

SUMMARIZE = "Summarize this review in 5 words: {review}"
COMBINE = "Combine these review summaries into one verdict: {summaries}"


class State(TypedDict):
    reviews: list[str]
    summaries: Annotated[list[str], operator.add]
    summary: str


def combine_prompt(summaries: list[str]) -> str:
    return COMBINE.format(summaries="; ".join(sorted(summaries)))   # sorted: same prompt whatever finished first


def mock_llm() -> MockLLM:
    # by_keyword, not a sequence: parallel branches can call the LLM in any order
    return MockLLM(by_keyword={
        "Combine": "Great battery and value, but the screen is too dim.",
        "battery": "Battery life is excellent",
        "screen": "Screen is too dim",
        "price": "Good value for the price",
    })


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def fan_out(state: State):
        if not state["reviews"]:
            return "combine"
        return [Send("summarize", {"review": r}) for r in state["reviews"]]

    def summarize(payload: dict):
        return {"summaries": [str(llm.invoke(SUMMARIZE.format(review=payload["review"])).content).strip()]}

    def combine(state: State):
        if not state["summaries"]:
            return {"summary": "No reviews."}
        return {"summary": str(llm.invoke(combine_prompt(state["summaries"])).content).strip()}

    g = StateGraph(State)
    g.add_node("summarize", summarize)
    g.add_node("combine", combine)
    g.add_conditional_edges(START, fan_out, ["summarize", "combine"])
    g.add_edge("summarize", "combine")
    g.add_edge("combine", END)
    return g.compile()


EXAMPLE_INPUT = {"reviews": ["The battery lasts two days, love it.",
                             "The screen is hard to read outdoors.",
                             "For this price it is a steal."],
                 "summaries": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
