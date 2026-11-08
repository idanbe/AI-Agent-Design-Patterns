"""09 · Map-reduce in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p09_map_reduce.graph import SUMMARIZE, combine_prompt, mock_llm, EXAMPLE_INPUT


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    reviews = inputs["reviews"]
    if not reviews:
        return {"summaries": [], "summary": "No reviews."}

    def summarize(review: str) -> str:
        return str(llm.invoke(SUMMARIZE.format(review=review)).content).strip()

    with ThreadPoolExecutor() as pool:                                   # map: all reviews at once
        summaries = list(pool.map(summarize, reviews))
    summary = str(llm.invoke(combine_prompt(summaries)).content).strip()  # reduce
    return {"summaries": summaries, "summary": summary}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
