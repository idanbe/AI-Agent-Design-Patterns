"""08 · Routing in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p08_routing.graph import CLASSIFY, HANDLER_PROMPTS, parse_route, mock_llm, EXAMPLE_INPUT


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    ticket = inputs["ticket"]
    route = parse_route(str(llm.invoke(CLASSIFY.format(ticket=ticket)).content))     # classify
    answer = str(llm.invoke(HANDLER_PROMPTS[route].format(ticket=ticket)).content)   # dict dispatch
    return {"route": route, "answer": answer}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
