"""16 · Supervisor in plain Python: same prompts, workers and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p16_supervisor.graph import FINISH, decide, work, final_answer, mock_llm, EXAMPLE_INPUT


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    task, history, turns = inputs["task"], [], 0
    while (choice := decide(task, history, turns, llm)) != FINISH:   # the boss picks the next worker
        history.append((choice, work(choice, task, history, llm)))   # that worker adds to the shared work
        turns += 1
    return {"answer": final_answer(history), "history": history, "turns": turns}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
