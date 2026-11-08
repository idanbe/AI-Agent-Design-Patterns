"""02 · ReWOO: plan every tool call up front with #E placeholders, run them without the LLM, then solve."""
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

PLANNER = """Make a plan to answer the task. For each step write a line "Plan: <reasoning>" and then
a line "#E<n> = <Tool>[<input>]". Tools: Search[query] looks up a fact; LLM[prompt] asks a language model.
A later input may use an earlier result by its variable, e.g. Search[capital of #E1].

Task: {task}"""

SOLVER = """Answer the task using the plan and the evidence below. Reply with the answer only.

Task: {task}
{evidence}"""

STEP_RE = re.compile(r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]")
FACTS = {"2018 fifa world cup winner": "France", "capital of france": "Paris"}
MOCK_PLAN = ("Plan: Find the 2018 World Cup winner.\n#E1 = Search[2018 FIFA World Cup winner]\n"
             "Plan: Find that country's capital.\n#E2 = Search[capital of #E1]")


class State(TypedDict):
    task: str
    plan: str
    steps: list[tuple[str, str, str]]
    results: dict[str, str]
    result: str


def parse_steps(plan: str) -> list[tuple[str, str, str]]:
    """(variable, tool, input) per plan line; text that doesn't match yields no steps."""
    return [(var, tool, arg) for _, var, tool, arg in STEP_RE.findall(plan)]


def search(query: str) -> str:
    return next((v for k, v in FACTS.items() if k in query.lower()), "unknown")


def run_steps(steps: list[tuple[str, str, str]], llm: BaseChatModel) -> dict[str, str]:
    """The worker: run each tool in order, filling in earlier #E results. No LLM reasoning in between."""
    results: dict[str, str] = {}
    for var, tool, arg in steps:
        for done in sorted(results, key=len, reverse=True):   # longest first: #E10 before #E1
            arg = arg.replace(done, results[done])
        if tool == "Search":
            results[var] = search(arg)
        elif tool == "LLM":
            results[var] = str(llm.invoke(arg).content)
        else:
            results[var] = "unknown tool"
    return results


def solver_prompt(task: str, plan: str, results: dict[str, str]) -> str:
    evidence = plan
    for var, value in results.items():
        evidence += f"\n{var} = {value}"
    return SOLVER.format(task=task, evidence=evidence)


def mock_llm() -> MockLLM:
    return MockLLM(responses=[MOCK_PLAN, "France won the 2018 World Cup; its capital is Paris."])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def plan(state: State):
        text = str(llm.invoke(PLANNER.format(task=state["task"])).content)
        return {"plan": text, "steps": parse_steps(text)}

    def work(state: State):
        return {"results": run_steps(state["steps"], llm)}

    def solve(state: State):
        prompt = solver_prompt(state["task"], state["plan"], state["results"])
        return {"result": str(llm.invoke(prompt).content)}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("work", work)
    g.add_node("solve", solve)
    g.add_edge(START, "plan")
    g.add_edge("plan", "work")
    g.add_edge("work", "solve")
    g.add_edge("solve", END)
    return g.compile()


EXAMPLE_INPUT = {"task": "What is the capital of the country that won the 2018 FIFA World Cup?"}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
