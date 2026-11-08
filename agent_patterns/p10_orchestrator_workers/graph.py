"""10 · Orchestrator-workers: an LLM decides the subtasks at runtime, workers do them in parallel, code merges."""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_SECTIONS = 5

# Must not contain the worker marker below — the mock tells the two prompts apart by it.
ORCHESTRATOR = """Plan the document below. List 2 to 5 section titles, one per line, nothing else.

Document: {task}"""

WORKER = """Write 1-2 sentences for one part of this document.
Document: {task}
Section: {section}
"""


class State(TypedDict):
    task: str
    sections: list[str]
    drafts: Annotated[list[tuple[str, str]], operator.add]
    report: str


def parse_sections(reply: str) -> list[str]:
    """Titles one per line; capped so a chatty model can't spawn dozens of workers."""
    return [line.strip("-• ").strip() for line in reply.splitlines() if line.strip()][:MAX_SECTIONS] or ["Overview"]


def assemble(sections: list[str], drafts: list[tuple[str, str]]) -> str:
    """Synthesizer: plain code. Uses the plan's order, not the order workers finished in."""
    texts = dict(drafts)
    return "\n\n".join(f"## {s}\n{texts[s]}" for s in sections)


def mock_llm() -> MockLLM:
    return MockLLM(responses=["Installation\nUsage\nLicense"], by_keyword={
        "Section: Installation": "Run `pip install todo-cli`.",
        "Section: Usage": 'Run `todo add "buy milk"`.',
        "Section: License": "MIT.",
    })


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def orchestrator(state: State):
        return {"sections": parse_sections(str(llm.invoke(ORCHESTRATOR.format(task=state["task"])).content))}

    def assign(state: State):
        return [Send("worker", {"task": state["task"], "section": s}) for s in state["sections"]]

    def worker(payload: dict):
        text = str(llm.invoke(WORKER.format(**payload)).content).strip()
        return {"drafts": [(payload["section"], text)]}

    def synthesize(state: State):
        return {"report": assemble(state["sections"], state["drafts"])}

    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("worker", worker)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", assign, ["worker"])
    g.add_edge("worker", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


EXAMPLE_INPUT = {"task": "Write a short README for todo-cli, a command-line todo app.", "drafts": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
