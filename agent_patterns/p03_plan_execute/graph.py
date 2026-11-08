"""03 · Plan-and-Execute: a planner writes steps, an executor does one, a replanner adjusts what is left."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_STEPS = 6
EMPTY_RESPONSE = "(empty reply)"  # never route on an empty string

PLANNER = """Make a short numbered plan for this objective, one step per line ("1. ...").

Objective: {input}"""

EXECUTOR = """Do this one step and reply with the result only.

Step: {step}
Done so far:
{done}"""

REPLANNER = """Objective: {input}

Done so far:
{done}

If the objective is complete, reply "FINAL: <answer>". Otherwise reply with the remaining
numbered steps only ("1. ...")."""


class State(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


def parse_plan(text: str) -> list[str]:
    return re.findall(r"^\s*\d+[.)]\s*(.+)$", text, re.M)


def done_text(past_steps: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {step}: {result}" for step, result in past_steps) or "(nothing yet)"


def replan(reply: str) -> dict:
    """FINAL: ends the run; a numbered list replaces the plan; anything else is taken as the answer."""
    if "FINAL:" in reply:
        return {"response": reply.split("FINAL:", 1)[1].strip() or EMPTY_RESPONSE}
    plan = parse_plan(reply)
    return {"plan": plan} if plan else {"response": reply.strip() or EMPTY_RESPONSE}


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "1. Look up the hotel price per night in Paris\n2. Add a daily food budget of 60 EUR",
        "Hotel: 150 EUR per night",
        "1. Add a daily food budget of 60 EUR",
        "150 + 60 = 210 EUR per day",
        "FINAL: Budget about 210 EUR per day (hotel 150 + food 60).",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def planner(state: State):
        reply = str(llm.invoke(PLANNER.format(input=state["input"])).content)
        return {"plan": parse_plan(reply) or [state["input"]]}

    def executor(state: State):
        step = state["plan"][0]
        reply = str(llm.invoke(EXECUTOR.format(step=step, done=done_text(state["past_steps"]))).content)
        return {"past_steps": [(step, reply)]}

    def replanner(state: State):
        if len(state["past_steps"]) >= MAX_STEPS:
            return {"response": f"Stopped after {MAX_STEPS} steps."}
        reply = str(llm.invoke(REPLANNER.format(input=state["input"], done=done_text(state["past_steps"]))).content)
        return replan(reply)

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("replanner", replanner)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "replanner")
    g.add_conditional_edges("replanner", lambda s: END if s.get("response") else "executor", ["executor", END])
    return g.compile()


EXAMPLE_INPUT = {"input": "Estimate a one-day budget in Paris: hotel plus food.", "past_steps": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
