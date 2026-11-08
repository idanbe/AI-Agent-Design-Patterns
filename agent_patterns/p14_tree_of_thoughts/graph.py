"""14 · Tree of Thoughts: propose several next steps, score each state, keep the best few, search deeper (BFS + beam)."""
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

BEAM = 2          # states kept per level
MAX_DEPTH = 3     # Game of 24 with 4 numbers needs exactly 3 steps
TARGET = 24
SCORES = {"impossible": 0.0, "sure": 1.0, "likely": 0.5}   # checked in this order
# ponytail: substring scoring reads "unlikely" as "likely"; parse a strict one-word reply if a real model says it

PROPOSE = ("Game of 24. Propose up to 3 next steps, one per line, as 'a op b = c' using two of the numbers.\n"
           "Numbers: {nums}\n")
EVALUATE = "Game of 24. Can these numbers reach 24? Answer sure, likely or impossible.\nRate: {nums}\n"

STEP_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([-+*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")   # "1. ", "2) ", "- ", "* "
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b}


class State(TypedDict):
    frontier: list[dict]
    candidates: list[dict]
    depth: int
    solution: list[str]


def fmt(nums: list[float]) -> str:
    return " ".join(f"{n:g}" for n in sorted(nums))


def apply_step(nums: list[float], line: str) -> dict | None:
    """Re-check one proposed step in code: both numbers must be available and the math must be right."""
    m = STEP_RE.match(LIST_MARKER.sub("", line, count=1))
    if not m:
        return None
    a, op, b, claimed = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
    rest = list(nums)
    for n in (a, b):
        if n not in rest:
            return None
        rest.remove(n)
    if op == "/" and b == 0:
        return None
    result = OPS[op](a, b)
    if abs(result - claimed) > 1e-6:
        return None
    return {"nums": rest + [result], "step": f"{a:g} {op} {b:g} = {result:g}"}


def score(reply: str) -> float:
    text = reply.lower()
    return next((v for word, v in SCORES.items() if word in text), 0.0)   # unreadable -> pruned


def expand(frontier: list[dict], llm: BaseChatModel) -> tuple[list[dict], list[str] | None]:
    """Thought generator: one PROPOSE call per state. Returns (new unique candidates, solution or None)."""
    candidates, seen = [], set()
    for thought in frontier:
        reply = str(llm.invoke(PROPOSE.format(nums=fmt(thought["nums"]))).content)
        for line in reply.splitlines():
            step = apply_step(thought["nums"], line)
            if step is None:
                continue
            child = {"nums": step["nums"], "steps": thought["steps"] + [step["step"]]}
            if len(child["nums"]) == 1 and abs(child["nums"][0] - TARGET) < 1e-6:
                return candidates, child["steps"]
            key = tuple(sorted(child["nums"]))
            if key not in seen:
                seen.add(key)
                candidates.append(child)
    return candidates, None


def score_and_prune(candidates: list[dict], llm: BaseChatModel) -> list[dict]:
    """State evaluator + search: score every candidate, drop the impossible ones, keep the best BEAM."""
    scored = [(score(str(llm.invoke(EVALUATE.format(nums=fmt(c["nums"]))).content)), c) for c in candidates]
    kept = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: -pair[0])   # stable sort
    return [c for _, c in kept[:BEAM]]


def mock_llm() -> MockLLM:
    return MockLLM(responses=["impossible"], by_keyword={
        "Numbers: 4 9 10 13\n": "10 - 4 = 6\n13 - 9 = 4\n4 + 9 = 13",
        "Numbers: 6 9 13\n": "13 - 9 = 4\n6 + 9 = 15",
        "Numbers: 4 4 10\n": "10 - 4 = 6\n4 * 4 = 16",
        "Numbers: 4 6\n": "6 * 4 = 24\n6 + 4 = 10",
        "Rate: 6 9 13\n": "likely",
        "Rate: 4 4 10\n": "likely",
        "Rate: 4 6\n": "sure",
    })


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def generate(state: State):
        candidates, solution = expand(state["frontier"], llm)
        update = {"candidates": candidates, "depth": state["depth"] + 1}
        return {**update, "solution": solution} if solution else update

    def evaluate(state: State):
        return {"frontier": score_and_prune(state["candidates"], llm)}

    g = StateGraph(State)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_edge(START, "generate")
    g.add_conditional_edges("generate", lambda s: END if s.get("solution") else "evaluate", ["evaluate", END])
    g.add_conditional_edges("evaluate",
                            lambda s: END if not s["frontier"] or s["depth"] >= MAX_DEPTH else "generate",
                            ["generate", END])
    return g.compile()


EXAMPLE_INPUT = {"frontier": [{"nums": [4, 9, 10, 13], "steps": []}], "depth": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
