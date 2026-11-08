"""Shared helpers: a scripted mock LLM (default), the OpenRouter switch, a demo runner and doc sync."""
import importlib
import os
import re
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-haiku-5.5"

PKG = Path(__file__).resolve().parent
PATTERNS: list[str] = sorted(p.parent.name for p in PKG.glob("p[0-9][0-9]_*/graph.py"))
MERMAID_BLOCK = re.compile(r"(<!-- mermaid:start -->).*?(<!-- mermaid:end -->)", re.S)


class MockLLM(BaseChatModel):
    """Scripted chat model: keyword replies first, then `responses` in order (cycling).

    Records every call so tests can assert cost (`calls`) and what the model saw (`prompts`).
    """

    responses: list[str | AIMessage] = []
    by_keyword: dict[str, str | AIMessage] = {}
    i: int = 0
    calls: int = 0
    prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = str(messages[-1].content)
        self.calls += 1
        self.prompts.append(text)
        reply = next((v for k, v in self.by_keyword.items() if k in text), None)
        if reply is None:
            if not self.responses:
                raise ValueError(f"MockLLM: no scripted reply for: {text[:80]}")
            reply = self.responses[self.i % len(self.responses)]
            self.i += 1
        msg = reply if isinstance(reply, AIMessage) else AIMessage(reply)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs) -> "MockLLM":
        return self  # ponytail: tool calls are scripted in `responses`, nothing to bind


def get_llm(mock: Callable[[], BaseChatModel]) -> BaseChatModel:
    """Mock by default; OpenRouter (OpenAI-compatible API) when USE_OPENROUTER=1."""
    if os.getenv("USE_OPENROUTER") != "1":
        return mock()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
                      base_url=OPENROUTER_BASE_URL, api_key=key, temperature=0)


def _message(m: BaseMessage) -> str:
    calls = getattr(m, "tool_calls", None)
    if calls:
        return "; ".join(f"{m.type} → {c['name']}({c['args']})" for c in calls)
    return f"{m.type}: {m.content}"


def _short(value, limit: int = 200) -> str:
    if isinstance(value, list) and value and all(isinstance(v, BaseMessage) for v in value):
        text = " | ".join(_message(v) for v in value)
    else:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def run_demo(graph, inputs: dict) -> dict:
    """Stream a graph, print one line per node update, return the final state."""
    final = {}
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue
        for node, update in chunk.items():
            print(f"▶ {node}")
            for key, value in (update or {}).items():
                print(f"    {key}: {_short(value)}")
    return final


def mermaid_for(name: str) -> str:
    mod = importlib.import_module(f"agent_patterns.{name}.graph")
    return mod.build_graph(mod.mock_llm()).get_graph().draw_mermaid()


def sync_docs() -> list[str]:
    """Rewrite the generated LangGraph diagram in every pattern README; return the names that changed."""
    changed = []
    for name in PATTERNS:
        readme = PKG / name / "README.md"
        old = readme.read_text()
        block = "\n```mermaid\n" + mermaid_for(name).strip() + "\n```\n"
        new = MERMAID_BLOCK.sub(lambda m: m.group(1) + block + m.group(2), old)
        if new != old:
            readme.write_text(new)
            changed.append(name)
    return changed


if __name__ == "__main__":
    print(sync_docs())
