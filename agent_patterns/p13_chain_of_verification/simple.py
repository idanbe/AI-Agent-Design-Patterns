"""13 · Chain-of-Verification in plain Python: same prompts and mock as graph.py; only the control flow differs."""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p13_chain_of_verification.graph import (BASELINE, PLAN, VERIFY, parse_checks, final_prompt,
                                                            mock_llm, EXAMPLE_INPUT)


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    question = inputs["question"]

    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    baseline = ask(BASELINE.format(question=question))                              # 1. draft
    checks = parse_checks(ask(PLAN.format(question=question, baseline=baseline)))   # 2. plan checks
    with ThreadPoolExecutor() as pool:                                              # 3. verify independently
        verified = list(zip(checks, pool.map(lambda q: ask(VERIFY.format(check=q)), checks)))
    final = ask(final_prompt(question, baseline, verified))                         # 4. revise
    return {"baseline": baseline, "checks": checks, "verified": verified, "final": final}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
