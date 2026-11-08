"""07 · Prompt chaining in plain Python: same prompts, gate and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p07_prompt_chaining.graph import (OUTLINE, WRITE, TRANSLATE, check_outline,
                                                      mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    outline = ask(OUTLINE.format(topic=inputs["topic"]))           # step 1
    if error := check_outline(outline):                             # gate: plain code
        return {"outline": outline, "error": error}
    draft = ask(WRITE.format(outline=outline))                      # step 2
    translation = ask(TRANSLATE.format(draft=draft))                # step 3
    return {"outline": outline, "draft": draft, "translation": translation}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
