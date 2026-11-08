"""05 · Self-Refine in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p05_self_refine.graph import (GENERATE, CRITIQUE, REFINE, is_done,
                                                  mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    task = inputs["task"]

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    draft, history, iterations = ask(GENERATE.format(task=task)), [], 0
    while True:
        feedback = ask(CRITIQUE.format(task=task, draft=draft))       # same model judges itself
        iterations += 1
        if is_done(feedback, iterations):
            return {"draft": draft, "history": history, "iterations": iterations}
        history.append(draft)
        draft = ask(REFINE.format(task=task, draft=draft, feedback=feedback))


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
