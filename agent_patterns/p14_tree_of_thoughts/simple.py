"""14 · Tree of Thoughts in plain Python: same prompts, checks and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p14_tree_of_thoughts.graph import MAX_DEPTH, expand, score_and_prune, mock_llm, EXAMPLE_INPUT


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    frontier = inputs["frontier"]
    for depth in range(1, MAX_DEPTH + 1):                 # breadth-first, one level per loop
        candidates, solution = expand(frontier, llm)       # propose thoughts
        if solution:
            return {"solution": solution, "depth": depth}
        frontier = score_and_prune(candidates, llm)        # score, prune, keep the beam
        if not frontier:
            break
    return {"solution": None, "depth": depth}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
