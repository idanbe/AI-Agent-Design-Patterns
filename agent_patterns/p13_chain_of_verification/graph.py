"""13 · Chain-of-Verification: draft, plan check questions, answer each independently, then revise the draft."""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_patterns.core import MockLLM, get_llm, run_demo

MAX_CHECKS = 5

BASELINE = "Answer the question: {question}"

PLAN = """Write one yes/no verification question per claim in this answer, one per line.
Question: {question}
Answer: {baseline}"""

# Factored verification: the verifier sees only its question, never the draft answer.
VERIFY = "Answer briefly and factually: {check}"

FINAL = """Revise the draft answer using the verified facts; drop anything they contradict.
Question: {question}
Draft: {baseline}
Verified facts:
{facts}
Reply with the revised answer only."""


class State(TypedDict):
    question: str
    baseline: str
    checks: list[str]
    verified: Annotated[list[tuple[str, str]], operator.add]
    final: str


def parse_checks(reply: str) -> list[str]:
    return [line.strip() for line in reply.splitlines() if line.strip().endswith("?")][:MAX_CHECKS]


def final_prompt(question: str, baseline: str, verified: list[tuple[str, str]]) -> str:
    facts = "\n".join(f"Q: {q} A: {a}" for q, a in sorted(verified)) or "(no checks)"   # sorted: order-independent
    return FINAL.format(question=question, baseline=baseline, facts=facts)


def mock_llm() -> MockLLM:
    # by_keyword, in this order: the verify calls run in parallel, so a sequence wouldn't be deterministic
    return MockLLM(by_keyword={
        "Revise": "Go and Dart. (Python was not created at Google.)",
        "verification question": "Was Go created at Google?\nWas Dart created at Google?\nWas Python created at Google?",
        "Was Go created": "Yes, at Google in 2009.",
        "Was Dart created": "Yes, at Google in 2011.",
        "Was Python created": "No, Guido van Rossum created it at CWI.",
        "List three": "Go, Dart and Python.",
    })


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    def draft_answer(state: State):   # node names can't reuse the state keys "baseline" / "final"
        return {"baseline": ask(BASELINE.format(question=state["question"]))}

    def plan_checks(state: State):
        return {"checks": parse_checks(ask(PLAN.format(question=state["question"], baseline=state["baseline"])))}

    def fan_out(state: State):
        return [Send("verify", {"check": q}) for q in state["checks"]] or "revise"

    def verify(payload: dict):
        return {"verified": [(payload["check"], ask(VERIFY.format(check=payload["check"])))]}

    def revise(state: State):
        return {"final": ask(final_prompt(state["question"], state["baseline"], state["verified"]))}

    g = StateGraph(State)
    g.add_node("draft_answer", draft_answer)
    g.add_node("plan_checks", plan_checks)
    g.add_node("verify", verify)
    g.add_node("revise", revise)
    g.add_edge(START, "draft_answer")
    g.add_edge("draft_answer", "plan_checks")
    g.add_conditional_edges("plan_checks", fan_out, ["verify", "revise"])
    g.add_edge("verify", "revise")
    g.add_edge("revise", END)
    return g.compile()


EXAMPLE_INPUT = {"question": "List three programming languages created at Google.", "verified": []}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
