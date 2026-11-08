"""06 · LLMCompiler: the planner emits a task graph; tasks whose inputs are ready run in parallel waves."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_patterns.core import MockLLM, get_llm, run_demo

PLANNER = """Plan tool calls to answer the question, one per line, as: <id>. <tool>(<input>)
Tools: search(query) looks up a number; subtract(a, b) returns a - b.
Use $<id> to pass an earlier task's result. Independent tasks will run in parallel.

Question: {question}"""

JOINER = """Answer the question from the task results. One sentence.

Question: {question}
Results:
{results}"""

TASK_RE = re.compile(r"^\s*(\d+)\.\s*(\w+)\((.*)\)\s*$", re.M)
FACTS = {"population of japan": "125", "population of germany": "84"}
MOCK_PLAN = '1. search("population of Japan")\n2. search("population of Germany")\n3. subtract($1, $2)'


class State(TypedDict):
    question: str
    tasks: dict[int, dict]
    results: Annotated[dict[int, str], operator.or_]
    waves: Annotated[list[list[int]], operator.add]
    answer: str


def parse_tasks(text: str) -> dict[int, dict]:
    tasks = {}
    for m in TASK_RE.finditer(text):
        arg = m.group(3).strip().strip("\"'")
        tasks[int(m.group(1))] = {"tool": m.group(2), "arg": arg,
                                  "deps": {int(d) for d in re.findall(r"\$(\d+)", arg)}}
    return tasks


def ready_tasks(tasks: dict[int, dict], results: dict[int, str]) -> list[int]:
    """Tasks not run yet whose dependencies all have results. Unresolvable deps never become ready."""
    return sorted(t for t, spec in tasks.items() if t not in results and spec["deps"] <= results.keys())


def search(query: str) -> str:
    return next((v for k, v in FACTS.items() if k in query.lower()), "unknown")


def subtract(arg: str) -> str:
    try:
        a, b = (float(x) for x in arg.split(","))
    except ValueError:
        return "error"
    return f"{a - b:g}"


TOOLS = {"search": search, "subtract": subtract}


def execute(task: dict, results: dict[int, str]) -> str:
    arg = task["arg"]
    for dep in sorted(results, key=lambda d: len(str(d)), reverse=True):   # $10 before $1
        arg = arg.replace(f"${dep}", results[dep])
    tool = TOOLS.get(task["tool"])
    return tool(arg) if tool else "unknown tool"


def joiner_prompt(question: str, tasks: dict[int, dict], results: dict[int, str]) -> str:
    lines = "\n".join(f"{t}. {tasks[t]['tool']}({tasks[t]['arg']}) = {results[t]}" for t in sorted(results))
    return JOINER.format(question=question, results=lines or "(no tasks)")


def mock_llm() -> MockLLM:
    return MockLLM(responses=[MOCK_PLAN, "Japan has the larger population, by about 41 million."])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def plan(state: State):
        return {"tasks": parse_tasks(str(llm.invoke(PLANNER.format(question=state["question"])).content))}

    def schedule(state: State):
        ready = ready_tasks(state["tasks"], state["results"])
        return {"waves": [ready]} if ready else {}

    def dispatch(state: State):
        ready = ready_tasks(state["tasks"], state["results"])
        if not ready:
            return "join"
        return [Send("run_task", {"id": t, "task": state["tasks"][t], "results": state["results"]}) for t in ready]

    def run_task(payload: dict):
        return {"results": {payload["id"]: execute(payload["task"], payload["results"])}}

    def join(state: State):
        # ponytail: the paper's joiner may also trigger a replan; skipped. Add a join -> plan edge if needed.
        prompt = joiner_prompt(state["question"], state["tasks"], state["results"])
        return {"answer": str(llm.invoke(prompt).content)}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("schedule", schedule)
    g.add_node("run_task", run_task)
    g.add_node("join", join)
    g.add_edge(START, "plan")
    g.add_edge("plan", "schedule")
    g.add_conditional_edges("schedule", dispatch, ["run_task", "join"])
    g.add_edge("run_task", "schedule")   # all tasks of a wave finish before the next schedule
    g.add_edge("join", END)
    return g.compile()


EXAMPLE_INPUT = {"question": "Which has more people, Japan or Germany, and by how many millions?",
                 "results": {}, "waves": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
