"""11 · Evaluator-optimizer in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p11_evaluator_optimizer.graph import (EVALUATE, MAX_ATTEMPTS, generate_prompt, parse_grade,
                                                          mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    task, feedback, attempts = inputs["task"], "", 0
    while True:
        draft = str(llm.invoke(generate_prompt(task, feedback)).content).strip()      # optimizer
        attempts += 1
        grade, feedback = parse_grade(str(llm.invoke(EVALUATE.format(task=task, draft=draft)).content))  # judge
        if grade == "pass" or attempts >= MAX_ATTEMPTS:
            return {"draft": draft, "grade": grade, "feedback": feedback, "attempts": attempts}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
