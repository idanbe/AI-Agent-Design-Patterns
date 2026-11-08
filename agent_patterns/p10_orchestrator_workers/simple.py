"""10 · Orchestrator-workers in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p10_orchestrator_workers.graph import (ORCHESTRATOR, WORKER, parse_sections, assemble,
                                                           mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    task = inputs["task"]
    sections = parse_sections(str(llm.invoke(ORCHESTRATOR.format(task=task)).content))   # LLM decides the split

    def write_section(section: str) -> tuple[str, str]:
        return section, str(llm.invoke(WORKER.format(task=task, section=section)).content).strip()

    with ThreadPoolExecutor() as pool:                                                  # workers in parallel
        drafts = list(pool.map(write_section, sections))
    return {"sections": sections, "drafts": drafts, "report": assemble(sections, drafts)}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT)["report"])
