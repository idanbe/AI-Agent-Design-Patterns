"""06 · LLMCompiler in plain Python: same prompts, tools and mock as graph.py; only the control flow differs."""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p06_llm_compiler.graph import (PLANNER, parse_tasks, ready_tasks, execute, joiner_prompt,
                                                   mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    question = inputs["question"]
    tasks = parse_tasks(str(llm.invoke(PLANNER.format(question=question)).content))
    results: dict[int, str] = {}
    waves: list[list[int]] = []
    with ThreadPoolExecutor() as pool:
        while ready := ready_tasks(tasks, results):                      # one wave = all tasks whose inputs exist
            outputs = pool.map(lambda t: execute(tasks[t], results), ready)
            results.update(zip(ready, outputs))
            waves.append(ready)
    answer = str(llm.invoke(joiner_prompt(question, tasks, results)).content)
    return {"tasks": tasks, "results": results, "waves": waves, "answer": answer}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
