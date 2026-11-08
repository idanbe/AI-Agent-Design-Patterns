"""03 · Plan-and-Execute in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p03_plan_execute.graph import (PLANNER, EXECUTOR, REPLANNER, MAX_STEPS, parse_plan,
                                                   done_text, replan, mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    goal = inputs["input"]
    plan = parse_plan(str(llm.invoke(PLANNER.format(input=goal)).content)) or [goal]
    past_steps: list[tuple[str, str]] = []
    while True:
        step = plan[0]                                                    # execute one step
        result = str(llm.invoke(EXECUTOR.format(step=step, done=done_text(past_steps))).content)
        past_steps.append((step, result))
        if len(past_steps) >= MAX_STEPS:
            return {"response": f"Stopped after {MAX_STEPS} steps.", "past_steps": past_steps}
        update = replan(str(llm.invoke(REPLANNER.format(input=goal, done=done_text(past_steps))).content))
        if "response" in update:
            return {"response": update["response"], "past_steps": past_steps}
        plan = update["plan"]                                             # replanned remaining steps


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
