"""15 · LATS (Language Agent Tree Search): Monte-Carlo tree search over real actions, LLM-scored, with reflections."""
import math
import operator
import re
from collections import Counter
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

GOAL = 10
ACTIONS = {"+1": lambda x: x + 1, "+3": lambda x: x + 3, "*2": lambda x: x * 2}   # the "environment"
MAX_DEPTH = 3          # steps allowed
MAX_ITERATIONS = 10    # select -> expand -> backprop rounds
EXPLORATION = 0.5      # UCT weight on exploring rarely visited branches
LAMBDA = 0.5           # value = LAMBDA * LLM score + (1 - LAMBDA) * self-consistency

PROPOSE = """Reach {goal} from {start} using +1, +3 or *2, at most {max_depth} steps.
{lessons}Path so far: {path}
Propose 3 next actions, one per line (you may repeat one you are sure about).
State: {state}
"""

EVALUATE = """Reach {goal} in at most {left} more steps using +1, +3 or *2. How promising is this state, 0-10?
Reply with a number.
Rate: {state}
"""

REFLECT = """This path failed: {path} ended at {value}, but the goal is {goal}.
In one sentence, what should the next attempt do differently?"""

ACTION_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[-•]\s+)?([+*])\s*(\d+)\s*$")


class Node:
    """One state in the search tree. `value` is a running mean of the rewards that passed through it."""

    def __init__(self, value: int, action: str | None = None, parent: "Node | None" = None):
        self.state, self.action, self.parent = value, action, parent
        self.children: list[Node] = []
        self.visits, self.value, self.dead = 0, 0.0, False
        self.depth = parent.depth + 1 if parent else 0

    def path(self) -> list[str]:
        node, steps = self, []
        while node.parent:
            steps.append(node.action)
            node = node.parent
        return steps[::-1]

    def __repr__(self) -> str:   # readable demo trace
        return f"Node({self.state}, path={self.path()}, V={self.value:.2f}, N={self.visits})"


class State(TypedDict):
    start: int
    root: Node           # ponytail: mutable tree in state; store a serializable dict if you add a checkpointer
    leaf: Node | None
    reward: float
    reflections: Annotated[list[str], operator.add]
    solution: list[str]
    iterations: int


def parse_actions(reply: str) -> list[str]:
    """Valid actions in the order proposed, repeats kept (they count as self-consistency votes)."""
    actions = []
    for line in reply.splitlines():
        m = ACTION_RE.match(line)
        if m and f"{m.group(1)}{m.group(2)}" in ACTIONS:
            actions.append(f"{m.group(1)}{m.group(2)}")
    return actions


def rate(reply: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", reply)
    return min(float(m.group()), 10.0) / 10 if m else 0.0       # unreadable -> 0


def uct(child: Node, parent_visits: int) -> float:
    return child.value + EXPLORATION * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)


def select(root: Node) -> Node | None:
    """1. Select: walk down by best UCT, skipping dead branches, to a node not expanded yet."""
    if root.dead:
        return None
    node = root
    while node.children:
        node = max((c for c in node.children if not c.dead), key=lambda c: uct(c, node.visits))
    return node


def prune_dead(node: Node | None) -> None:
    """A node whose children are all dead ends is a dead end too."""
    while node and node.children and all(c.dead for c in node.children):
        node.dead = True
        node = node.parent


def expand(leaf: Node, llm: BaseChatModel, reflections: list[str], start: int) -> tuple[list[str] | None, list[str], float]:
    """2-3. Expand + evaluate: sample actions, run them, score children. Failed paths get a reflection.
    Returns (solution or None, new reflections, best child value)."""
    lessons = "Lessons from failed paths:\n" + "\n".join(f"- {r}" for r in reflections) + "\n" if reflections else ""
    reply = str(llm.invoke(PROPOSE.format(goal=GOAL, start=start, max_depth=MAX_DEPTH, lessons=lessons,
                                          path=leaf.path(), state=leaf.state)).content)
    samples = parse_actions(reply)
    if not samples:                              # nothing usable -> dead branch
        leaf.dead = True
        prune_dead(leaf.parent)
        return None, [], 0.0
    new_lessons, best = [], 0.0
    for action, votes in Counter(samples).items():
        child = Node(ACTIONS[action](leaf.state), action, leaf)
        leaf.children.append(child)
        child.visits = 1
        if child.state == GOAL:
            return child.path(), new_lessons, 1.0
        if child.state > GOAL or child.depth >= MAX_DEPTH:      # the environment says: failed
            child.dead = True                    # ponytail: no random rollout; failure is known right away here
            reflect = REFLECT.format(path=child.path(), value=child.state, goal=GOAL)
            new_lessons.append(str(llm.invoke(reflect).content).strip())
            continue
        score = rate(str(llm.invoke(EVALUATE.format(goal=GOAL, left=MAX_DEPTH - child.depth,
                                                     state=child.state)).content))
        child.value = LAMBDA * score + (1 - LAMBDA) * votes / len(samples)
        best = max(best, child.value)
    prune_dead(leaf)
    return None, new_lessons, best


def backprop(node: Node, reward: float) -> None:
    """5. Backpropagate: every node on the path gets one more visit and moves its mean toward the reward."""
    while node:
        node.visits += 1
        node.value += (reward - node.value) / node.visits
        node = node.parent


def mock_llm() -> MockLLM:
    return MockLLM(by_keyword={
        "This path failed": "Doubling 7 overshoots 10; from 4 try +1 first, then double 5.",
        "Rate: 4\n": "7", "Rate: 2\n": "1", "Rate: 7\n": "8", "Rate: 5\n": "6",
        "State: 1\n": "+3\n+3\n*2",
        "State: 4\n": "+3\n+3\n+1",
        "State: 7\n": "*2\n*2\n*2",
        "State: 5\n": "*2\n*2\n+3",
    })


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def select_node(state: State):
        root = state.get("root") or Node(state["start"])
        return {"root": root, "leaf": select(root), "iterations": state["iterations"] + 1}

    def expand_node(state: State):
        if state["leaf"] is None:
            return {}
        solution, lessons, reward = expand(state["leaf"], llm, state["reflections"], state["start"])
        update = {"reflections": lessons, "reward": reward}
        return {**update, "solution": solution} if solution else update

    def backprop_node(state: State):
        if state["leaf"] is not None and not state.get("solution"):
            backprop(state["leaf"], state["reward"])
        return {}

    def route(state: State):
        done = state.get("solution") or state["root"].dead or state["iterations"] >= MAX_ITERATIONS
        return END if done else "select"

    g = StateGraph(State)
    g.add_node("select", select_node)
    g.add_node("expand", expand_node)
    g.add_node("backprop", backprop_node)
    g.add_edge(START, "select")
    g.add_edge("select", "expand")
    g.add_edge("expand", "backprop")
    g.add_conditional_edges("backprop", route, ["select", END])
    return g.compile()


EXAMPLE_INPUT = {"start": 1, "reflections": [], "iterations": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
