"""04 · Reflexion: try, get an external verdict, write a lesson, retry with all lessons in memory."""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_TRIALS = 3

ACTOR = """{task}
{lessons}
Reply with the slogan only."""

REFLECT = """Your slogan failed these checks: {feedback}
Slogan: {attempt}
In one or two sentences, say what went wrong and what to do differently next time."""


class State(TypedDict):
    task: str
    attempt: str
    feedback: list[str]
    reflections: Annotated[list[str], operator.add]
    trials: int
    passed: bool


def check_slogan(text: str) -> list[str]:
    """The external evaluator: plain rules, no LLM. Returns the failed checks."""
    failures = []
    words = len(text.split())
    if words > 8:
        failures.append(f"too long ({words} words, max 8)")
    if "planet" not in text.lower():
        failures.append("must mention 'planet'")
    if "!" in text:
        failures.append("no exclamation marks")
    return failures


def actor_prompt(task: str, reflections: list[str]) -> str:
    lessons = "Lessons from earlier attempts:\n" + "\n".join(f"- {r}" for r in reflections) if reflections else ""
    return ACTOR.format(task=task, lessons=lessons)


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "Stay hydrated and save the planet every single day!",
        "I wrote 9 words and used '!'. Next time: at most 8 words, no exclamation marks.",
        "Refill daily, protect the planet.",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def actor(state: State):
        attempt = str(llm.invoke(actor_prompt(state["task"], state["reflections"])).content).strip()
        return {"attempt": attempt, "trials": state["trials"] + 1}

    def evaluate(state: State):
        feedback = check_slogan(state["attempt"])
        return {"feedback": feedback, "passed": not feedback}

    def reflect(state: State):
        prompt = REFLECT.format(feedback="; ".join(state["feedback"]), attempt=state["attempt"])
        return {"reflections": [str(llm.invoke(prompt).content).strip()]}

    def route(state: State):
        return END if state["passed"] or state["trials"] >= MAX_TRIALS else "reflect"

    g = StateGraph(State)
    g.add_node("actor", actor)
    g.add_node("evaluate", evaluate)
    g.add_node("reflect", reflect)
    g.add_edge(START, "actor")
    g.add_edge("actor", "evaluate")
    g.add_conditional_edges("evaluate", route, ["reflect", END])
    g.add_edge("reflect", "actor")
    return g.compile()


EXAMPLE_INPUT = {"task": "Write a slogan for a reusable water bottle. Max 8 words, mention 'planet', "
                         "no exclamation marks.",
                 "trials": 0, "reflections": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
