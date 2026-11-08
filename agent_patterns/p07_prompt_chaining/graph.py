"""07 · Prompt chaining: a fixed sequence of LLM calls, with a plain-code gate checking the output between them."""
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

BULLETS_NEEDED = 3

OUTLINE = """Write exactly 3 selling points for: {topic}
One per line, each line starting with "- ". No other text."""

WRITE = """Turn these selling points into one short, friendly product paragraph:
{outline}"""

TRANSLATE = """Translate to Hebrew. Reply with the translation only:
{draft}"""


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    translation: str
    error: str


def check_outline(outline: str) -> str:
    """The gate: plain code, no LLM. Returns an error message, or "" if the outline is good."""
    bullets = sum(1 for line in outline.splitlines() if line.strip().startswith("- "))
    return "" if bullets >= BULLETS_NEEDED else f"outline has {bullets} bullet(s), need {BULLETS_NEEDED}"


def mock_llm() -> MockLLM:
    return MockLLM(responses=[
        "- Keeps drinks cold 24h\n- Made from recycled steel\n- Lifetime warranty",
        "Meet the bottle that keeps drinks cold for 24 hours, is made from recycled steel "
        "and comes with a lifetime warranty.",
        "הכירו את הבקבוק ששומר על משקאות קרים 24 שעות, עשוי מפלדה ממוחזרת ומגיע עם אחריות לכל החיים.",
    ])


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    def make_outline(state: State):  # can't be named "outline": that is a state key
        return {"outline": ask(OUTLINE.format(topic=state["topic"]))}

    def gate(state: State):
        error = check_outline(state["outline"])
        return {"error": error} if error else {}

    def write(state: State):
        return {"draft": ask(WRITE.format(outline=state["outline"]))}

    def translate(state: State):
        return {"translation": ask(TRANSLATE.format(draft=state["draft"]))}

    g = StateGraph(State)
    g.add_node("make_outline", make_outline)
    g.add_node("gate", gate)
    g.add_node("write", write)
    g.add_node("translate", translate)
    g.add_edge(START, "make_outline")
    g.add_edge("make_outline", "gate")
    g.add_conditional_edges("gate", lambda s: END if s.get("error") else "write", ["write", END])
    g.add_edge("write", "translate")
    g.add_edge("translate", END)
    return g.compile()


EXAMPLE_INPUT = {"topic": "product blurb for an insulated steel water bottle"}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
