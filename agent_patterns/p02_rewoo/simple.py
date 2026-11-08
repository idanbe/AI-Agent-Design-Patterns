"""02 · ReWOO in plain Python: same prompts, tools and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p02_rewoo.graph import (PLANNER, parse_steps, run_steps, solver_prompt,
                                            mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    plan = str(llm.invoke(PLANNER.format(task=inputs["task"])).content)     # 1 LLM call: plan everything
    results = run_steps(parse_steps(plan), llm)                              # tools only, no LLM reasoning
    result = str(llm.invoke(solver_prompt(inputs["task"], plan, results)).content)  # 1 LLM call: solve
    return {"plan": plan, "results": results, "result": result}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
