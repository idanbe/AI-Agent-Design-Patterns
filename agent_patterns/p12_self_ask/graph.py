"""12 · Self-Ask: the model asks itself follow-up questions, a search tool answers each, then it composes the answer."""
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_HOPS = 4
FOLLOW_UP = "Follow up:"
FINAL = "So the final answer is:"
LIMIT_ANSWER = "unknown (follow-up limit reached)"

ASK = """Answer the question by asking yourself simpler follow-up questions.
Use exactly this format, one line at a time:
Follow up: <a simpler question>
Intermediate answer: <filled in by a search tool - never write this line yourself>
So the final answer is: <answer>

Write ONLY the next line.

Question: {question}
Are follow up questions needed here: Yes.{scratchpad}"""

FACTS = {"eiffel tower": "1889", "president of the united states in 1889": "Benjamin Harrison"}


class State(TypedDict):
    question: str
    scratchpad: str
    pending: str
    hops: int
    answer: str


def search(question: str) -> str:
    return next((v for k, v in FACTS.items() if k in question.lower()), "unknown")


def read_line(reply: str) -> tuple[str, str]:
    """('answer', text) or ('follow_up', question). Text in neither format is taken as the answer."""
    if FINAL in reply:
        return "answer", reply.split(FINAL, 1)[1].strip()
    if FOLLOW_UP in reply:
        return "follow_up", reply.split(FOLLOW_UP, 1)[1].strip().splitlines()[0].strip()
    return "answer", reply.strip()


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "Follow up: When was the Eiffel Tower completed?",
        "Follow up: Who was president of the United States in 1889?",
        "So the final answer is: Benjamin Harrison",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def ask(state: State):
        if state["hops"] >= MAX_HOPS:
            return {"answer": LIMIT_ANSWER}
        reply = str(llm.invoke(ASK.format(question=state["question"], scratchpad=state["scratchpad"])).content)
        kind, text = read_line(reply)
        if kind == "answer":
            return {"answer": text}
        return {"scratchpad": state["scratchpad"] + f"\n{FOLLOW_UP} {text}", "pending": text,
                "hops": state["hops"] + 1}

    def search_node(state: State):
        return {"scratchpad": state["scratchpad"] + f"\nIntermediate answer: {search(state['pending'])}"}

    g = StateGraph(State)
    g.add_node("ask", ask)
    g.add_node("search", search_node)
    g.add_edge(START, "ask")
    g.add_conditional_edges("ask", lambda s: END if s.get("answer") else "search", ["search", END])
    g.add_edge("search", "ask")
    return g.compile()


EXAMPLE_INPUT = {"question": "Who was president of the United States when the Eiffel Tower was completed?",
                 "scratchpad": "", "hops": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
