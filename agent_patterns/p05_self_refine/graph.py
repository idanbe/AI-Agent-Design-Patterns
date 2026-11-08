"""05 · Self-Refine: one model drafts, critiques its own draft, and rewrites it until it finds no issues."""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_ITERATIONS = 3
STOP_WORD = "NO_ISSUES"

GENERATE = "{task}\nReply with the result only."

CRITIQUE = """Review this draft for the task below. List concrete problems in one or two lines.
If it is already good, reply exactly NO_ISSUES.

Task: {task}
Draft: {draft}"""

REFINE = """Rewrite the draft to fix the feedback. Reply with the new draft only.

Task: {task}
Draft: {draft}
Feedback: {feedback}"""


class State(TypedDict):
    task: str
    draft: str
    feedback: str
    history: Annotated[list[str], operator.add]
    iterations: int


def is_done(feedback: str, iterations: int) -> bool:
    return STOP_WORD in feedback or iterations >= MAX_ITERATIONS


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "fixed bug",
        "Too vague: say what was fixed and where.",
        "fix(profile): handle users without an email address",
        "NO_ISSUES",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    def generate(state: State):
        return {"draft": ask(GENERATE.format(task=state["task"]))}

    def critique(state: State):  # node can't be named "feedback": that is a state key
        feedback = ask(CRITIQUE.format(task=state["task"], draft=state["draft"]))
        return {"feedback": feedback, "iterations": state["iterations"] + 1}

    def refine(state: State):
        draft = ask(REFINE.format(task=state["task"], draft=state["draft"], feedback=state["feedback"]))
        return {"draft": draft, "history": [state["draft"]]}

    g = StateGraph(State)
    g.add_node("generate", generate)
    g.add_node("critique", critique)
    g.add_node("refine", refine)
    g.add_edge(START, "generate")
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", lambda s: END if is_done(s["feedback"], s["iterations"]) else "refine",
                            ["refine", END])
    g.add_edge("refine", "critique")
    return g.compile()


EXAMPLE_INPUT = {"task": "Write a git commit message for: fixed a crash when a user has no email",
                 "iterations": 0, "history": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
