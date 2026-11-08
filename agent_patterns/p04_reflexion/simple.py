"""04 · Reflexion in plain Python: same prompts, evaluator and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p04_reflexion.graph import (REFLECT, MAX_TRIALS, actor_prompt, check_slogan,
                                                mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    reflections: list[str] = []                                            # episodic memory
    for trial in range(1, MAX_TRIALS + 1):
        attempt = str(llm.invoke(actor_prompt(inputs["task"], reflections)).content).strip()
        feedback = check_slogan(attempt)                                   # external evaluator
        if not feedback or trial == MAX_TRIALS:
            return {"attempt": attempt, "passed": not feedback, "trials": trial, "reflections": reflections}
        prompt = REFLECT.format(feedback="; ".join(feedback), attempt=attempt)
        reflections.append(str(llm.invoke(prompt).content).strip())        # verbal lesson


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
