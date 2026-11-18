"""16 · Supervisor: a "boss" LLM reads the shared work and picks which specialist agent acts next, until FINISH."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

FINISH = "FINISH"
MAX_TURNS = 6                 # worker runs; the supervisor stops itself after this many
NO_ANSWER = "(no answer)"

SUPERVISOR = """You manage a team: researcher (finds facts), analyst (does the math), writer (writes the final answer).
Read the work so far and reply with ONE word: the worker who should act next, or FINISH when the task is done.

Task: {task}
Work so far:
{work}"""

# Each worker has its own small prompt (and in real systems, its own tools) - that is the point of the pattern.
WORKER_PROMPTS = {
    "researcher": "You are the researcher. Find the facts the task needs. Reply with the facts only.",
    "analyst": "You are the analyst. Do the calculation the task needs from the facts so far. Reply with the result.",
    "writer": "You are the writer. Write the final answer for the user from the work so far. One sentence.",
}

WORKER = """{role}

Task: {task}
Work so far:
{work}"""


class State(TypedDict):
    task: str
    history: Annotated[list[tuple[str, str]], operator.add]   # (worker, output) - the shared work
    next_worker: str
    turns: int
    answer: str


def work_text(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in history) or "(nothing yet)"


def parse_next(reply: str) -> str:
    """Worker name, or FINISH. Anything unrecognized stops the run instead of looping on a guess."""
    word = re.sub(r"[^a-z]", "", reply.lower())
    return word if word in WORKER_PROMPTS else FINISH


def decide(task: str, history: list[tuple[str, str]], turns: int, llm: BaseChatModel) -> str:
    """The supervisor's choice. At the turn cap it stops without asking the LLM."""
    if turns >= MAX_TURNS:
        return FINISH
    return parse_next(str(llm.invoke(SUPERVISOR.format(task=task, work=work_text(history))).content))


def work(worker: str, task: str, history: list[tuple[str, str]], llm: BaseChatModel) -> str:
    prompt = WORKER.format(role=WORKER_PROMPTS[worker], task=task, work=work_text(history))
    return str(llm.invoke(prompt).content).strip() or "(empty reply)"


def final_answer(history: list[tuple[str, str]]) -> str:
    return history[-1][1] if history else NO_ANSWER


def mock_llm() -> MockLLM:
    # supervisor and workers take turns, so a plain sequence is deterministic here
    return MockLLM(responses=[
        "researcher", "Q2 revenue $80k, Q3 revenue $120k",
        "analyst", "Growth = (120-80)/80 = 50%",
        "writer", "Summary: Q3 revenue grew 50% QoQ, from $80k to $120k.",
        "FINISH",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def supervisor(state: State):
        choice = decide(state["task"], state["history"], state["turns"], llm)
        if choice == FINISH:
            return {"next_worker": FINISH, "answer": final_answer(state["history"])}
        return {"next_worker": choice}

    def make_worker(name: str):
        def worker(state: State):
            output = work(name, state["task"], state["history"], llm)
            return {"history": [(name, output)], "turns": state["turns"] + 1}
        return worker

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for name in WORKER_PROMPTS:
        g.add_node(name, make_worker(name))
        g.add_edge(name, "supervisor")          # every worker reports back to the boss
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor",
                            lambda s: END if s["next_worker"] == FINISH else s["next_worker"],
                            {**{name: name for name in WORKER_PROMPTS}, END: END})
    return g.compile()


EXAMPLE_INPUT = {"task": "Write a one-line summary of our Q3 revenue growth.", "history": [], "turns": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
