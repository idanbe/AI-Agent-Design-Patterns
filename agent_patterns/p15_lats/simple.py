"""15 · LATS in plain Python: same prompts, tree helpers and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p15_lats.graph import (MAX_ITERATIONS, Node, select, expand, backprop,
                                           mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    root, reflections, iterations = Node(inputs["start"]), [], 0
    while True:
        iterations += 1
        leaf = select(root)                                    # 1. select by UCT
        if leaf is not None:
            solution, lessons, reward = expand(leaf, llm, reflections, inputs["start"])  # 2-3. expand + evaluate
            reflections += lessons                             # 4. reflections from failed paths
            if solution:
                return {"solution": solution, "reflections": reflections, "iterations": iterations}
            backprop(leaf, reward)                             # 5. backpropagate
        if root.dead or iterations >= MAX_ITERATIONS:          # same stop check as the graph's router
            return {"solution": None, "reflections": reflections, "iterations": iterations}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
