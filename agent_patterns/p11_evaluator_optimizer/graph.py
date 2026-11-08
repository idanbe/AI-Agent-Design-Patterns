"""11 · Evaluator-optimizer: a generator writes, a separate judge grades PASS/FAIL with feedback, loop until PASS."""
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_ATTEMPTS = 3

GENERATE = """{task}
{feedback}
Reply with the translation only."""

EVALUATE = """You are a strict reviewer. Grade the translation against all criteria:
1. Correct meaning  2. Warm and friendly tone  3. Natural, fluent Hebrew
Reply "PASS" if all are met, otherwise "FAIL: <what to fix>".

Task: {task}
Translation: {draft}"""


class State(TypedDict):
    task: str
    draft: str
    grade: str
    feedback: str
    attempts: int


def generate_prompt(task: str, feedback: str) -> str:
    return GENERATE.format(task=task, feedback=f"Feedback on the last attempt: {feedback}" if feedback else "")


def parse_grade(reply: str) -> tuple[str, str]:
    """(grade, feedback). Anything that isn't a clear PASS counts as a fail."""
    text = reply.strip()
    if re.match(r"\W*PASS\b", text, re.I):          # "**PASS**" passes, "Passable, but..." does not
        return "pass", ""
    return "fail", text.split("FAIL:", 1)[1].strip() if "FAIL:" in text else text


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "הזמנתך נשלחה.",
        "FAIL: Correct but cold and formal; make it warm and friendly.",
        "איזה כיף, ההזמנה שלך כבר בדרך אליך!",
        "PASS",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def generate(state: State):
        draft = str(llm.invoke(generate_prompt(state["task"], state["feedback"])).content).strip()
        return {"draft": draft, "attempts": state["attempts"] + 1}

    def evaluate(state: State):
        # ponytail: judge shares the generator's llm; give it its own (stronger) model if needed
        grade, feedback = parse_grade(str(llm.invoke(EVALUATE.format(task=state["task"], draft=state["draft"])).content))
        return {"grade": grade, "feedback": feedback}

    def route(state: State):
        return END if state["grade"] == "pass" or state["attempts"] >= MAX_ATTEMPTS else "generate"

    g = StateGraph(State)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_edge(START, "generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges("evaluate", route, ["generate", END])
    return g.compile()


EXAMPLE_INPUT = {"task": "Translate to Hebrew in a warm, friendly tone: 'Your order has shipped.'",
                 "attempts": 0, "feedback": ""}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
