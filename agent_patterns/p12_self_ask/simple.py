"""12 · Self-Ask in plain Python: same prompt, search tool and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p12_self_ask.graph import (ASK, FOLLOW_UP, MAX_HOPS, LIMIT_ANSWER, read_line, search,
                                               mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    scratchpad = ""
    for _ in range(MAX_HOPS):
        reply = str(llm.invoke(ASK.format(question=inputs["question"], scratchpad=scratchpad)).content)
        kind, text = read_line(reply)
        if kind == "answer":
            return {"answer": text, "scratchpad": scratchpad}
        scratchpad += f"\n{FOLLOW_UP} {text}"                               # the model's own sub-question
        scratchpad += f"\nIntermediate answer: {search(text)}"              # answered by the tool
    return {"answer": LIMIT_ANSWER, "scratchpad": scratchpad}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
