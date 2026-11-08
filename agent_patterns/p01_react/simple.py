"""01 · ReAct in plain Python: same prompt, tools and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage

from agent_patterns.core import get_llm
from agent_patterns.p01_react.graph import SYSTEM, TOOLS, mock_llm, EXAMPLE_INPUT

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
MAX_TURNS = 10  # same job as LangGraph's recursion_limit


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    model = llm.bind_tools(TOOLS)
    messages = [SystemMessage(SYSTEM)] + list(inputs["messages"])
    observations = []
    for _ in range(MAX_TURNS):
        ai = model.invoke(messages)                      # think
        messages.append(ai)
        if not ai.tool_calls:                            # no action -> final answer
            return {"answer": ai.content, "observations": observations}
        for call in ai.tool_calls:                       # act
            result = str(TOOLS_BY_NAME[call["name"]].invoke(call["args"]))
            observations.append(result)                  # observe
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
    return {"answer": "Stopped: turn limit reached.", "observations": observations}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
