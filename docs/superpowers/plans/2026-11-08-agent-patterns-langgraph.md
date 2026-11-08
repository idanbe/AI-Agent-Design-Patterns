# Agent Patterns in LangGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new repo, `agent-patterns-langgraph`, that teaches 14 AI-agent design patterns as small runnable LangGraph graphs. Each pattern gets its own doc, and the README carries a comparison table and a "which pattern should I use?" guide. A repeatable recipe lets the user add one new pattern per day.

**Architecture:** Each pattern is a self-contained folder, `agent_patterns/pNN_name/`. Its `graph.py` exposes `build_graph(llm)`, `mock_llm()` and `EXAMPLE_INPUT`. Its `simple.py` runs the same pattern in plain Python (`run(inputs, llm)`). It reuses `graph.py`'s prompts, tools, parsers and mock, so only the control flow differs, and a parity test keeps the two versions in step. Its `test_graph.py` holds the tests. Its `README.md` gives a plain-words explanation, a hand-drawn concept diagram, and the Mermaid diagram LangGraph draws from the compiled graph. The shared `agent_patterns/core.py` holds a scripted `MockLLM`, which is the default: it runs offline, is deterministic and counts its calls. It also holds `get_llm()`, which switches to OpenRouter when `USE_OPENROUTER=1`. A conventions test keeps the folders, diagrams and root README in sync, and a repo-local `/add-pattern` Claude Code skill holds the daily recipe.

**Tech Stack:** Python 3.12, uv, langgraph 1.2.x, langchain-core 1.7.x, langchain-openai 1.7.x (`ChatOpenAI` pointed at OpenRouter's OpenAI-compatible API), pytest 9.

**Spec:** There is no separate spec doc. The requirements below come from the user's chat on 2026-11-08 and from the curriculum of the routine "⚡ Daily agent pattern lesson" (`trig_013zBrjiHbQcMJU8Pwn2zv5T`, items 0–13).

## Requirements

- R1: New git repo at `/home/idanbe/Desktop/ai-projects/agent-patterns-langgraph`.
- R2: The patterns are routine curriculum items 0–13, numbered 01–14 in the repo (see Pattern list).
- R3: Each pattern is its own folder with a LangGraph implementation, a runnable example, and a `README.md`. The README has a diagram, a simple explanation, and when to use it and when not to (user follow-up, 2026-11-08).
- R4: The README has a comparison table across all patterns and a decision guide.
- R5: The LLM is a mock by default. OpenRouter is used when the user enables it.
- R6: The user adds one pattern per day, so adding one is a repeatable recipe (`/add-pattern`) that tests enforce.
- R7: Each pattern also has a plain-Python version next to the LangGraph one, so readers see what LangGraph adds and when plain code is enough (user follow-up, 2026-11-08).

## Pattern list (paper IDs verified against the arXiv API on 2026-11-08)

| NN | module | Name | Family | Source |
|---|---|---|---|---|
| 01 | `p01_react` | ReAct | Tool agent | arXiv 2210.03629, Yao et al., 2022 |
| 02 | `p02_rewoo` | ReWOO | Tool agent | arXiv 2305.18323, Xu et al., 2023 |
| 03 | `p03_plan_execute` | Plan-and-Execute | Tool agent | arXiv 2305.04091 (Plan-and-Solve), Wang et al., 2023 |
| 04 | `p04_reflexion` | Reflexion | Self-correction | arXiv 2303.11366, Shinn et al., 2023 |
| 05 | `p05_self_refine` | Self-Refine | Self-correction | arXiv 2303.17651, Madaan et al., 2023 |
| 06 | `p06_llm_compiler` | LLMCompiler | Tool agent | arXiv 2312.04511, Kim et al., 2023 |
| 07 | `p07_prompt_chaining` | Prompt chaining | Workflow | Anthropic, "Building effective agents" (Schluntz & Zhang, Dec 2024) |
| 08 | `p08_routing` | Routing | Workflow | same |
| 09 | `p09_map_reduce` | Parallelization (map-reduce) | Workflow | same |
| 10 | `p10_orchestrator_workers` | Orchestrator-workers | Workflow | same |
| 11 | `p11_evaluator_optimizer` | Evaluator-optimizer | Workflow | same |
| 12 | `p12_self_ask` | Self-Ask | Tool agent | arXiv 2210.03350, Press et al., 2022 |
| 13 | `p13_chain_of_verification` | Chain-of-Verification | Self-correction | arXiv 2309.11495, Dhuliawala et al., 2023 |
| 14 | `p14_tree_of_thoughts` | Tree of Thoughts | Search | arXiv 2305.10601, Yao et al., 2023 |

Anthropic URL (HTTP 200 verified): `https://www.anthropic.com/engineering/building-effective-agents`

## Global Constraints

- Python `>=3.12`, managed with uv. Runtime deps are exactly `langgraph>=1.2`, `langchain-core>=1.7` and `langchain-openai>=1.7`. The only dev dep is `pytest>=9`. No python-dotenv: use `uv run --env-file .env`.
- The mock is the default. A real model is used only when `USE_OPENROUTER=1`. Key: `OPENROUTER_API_KEY`. Model: `OPENROUTER_MODEL`, default `anthropic/claude-haiku-5.5`. Base URL: `https://openrouter.ai/api/v1`. `temperature=0`.
- Tests never touch the network. Every pattern test passes `mock_llm()` or a `MockLLM` explicitly. Only `tests/test_openrouter_smoke.py` may call out, and it skips when no key is set.
- Module contract: `build_graph(llm: BaseChatModel | None = None) -> CompiledStateGraph` (if `llm is None`, it uses `get_llm(mock_llm)`), `mock_llm() -> MockLLM`, and `EXAMPLE_INPUT: dict` holding every state key the graph reads, including empty reducer lists. `if __name__ == "__main__": run_demo(build_graph(), EXAMPLE_INPUT)`.
- Names: folder `agent_patterns/pNN_snake/` containing `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py` and `README.md`. NN is consecutive from 01. Run a pattern with `python -m agent_patterns.pNN_snake.graph`.
- Diagram labels (GitHub clips long labels): always wrap node text in double quotes (`A["Run tool"]`, `D{"Passed?"}`), because `(`, `:` and `#` break the parser otherwise. At most 20 characters per line, with `<br/>` for a second line. No arrows or emoji inside labels. Edge labels are 1–2 words. Check every hand-drawn diagram with the Mermaid validator tool if one is available, otherwise at https://mermaid.live.
- Folder README language: the `## In plain words` section uses no jargon (no "DAG", "reducer" or "superstep") and includes one everyday analogy.
- Plain-Python contract: `simple.py` defines `run(inputs: dict, llm: BaseChatModel | None = None) -> dict`, with `llm = get_llm(mock_llm) if llm is None else llm`, and `__main__` prints `run(EXAMPLE_INPUT)`. It imports prompts, tools, parsers, caps and `mock_llm` from its own `graph.py` and never re-implements them. It never imports `langgraph`; `langchain_core` message types are allowed. Parallel steps use `concurrent.futures.ThreadPoolExecutor`. Its result keys use the same names as the graph state.
- `graph.py` nodes are thin wrappers around module-level prompts, tools, parsers and helpers, so `simple.py` can reuse them.
- Prompts are module-level `UPPER_CASE` str constants with `.format()` fields, and each one spells out the exact reply format it expects.
- Parsers never raise on unexpected model text. Each has a documented fallback.
- Every loop has an explicit cap constant. The one exception is ReAct, which relies on LangGraph's `recursion_limit`.
- Every `add_conditional_edges` call passes an explicit path list or map, so `draw_mermaid()` draws those edges, including the `Send` fan-outs.
- Reading an optional state key uses `state.get(...)`.
- Never `eval` or `exec` model output.
- Never invent sources. Before committing, check every doc link with `curl -sIL -o /dev/null -w "%{http_code}" <url>` and expect `200`.
- Docs are in English, at most ~80 lines per pattern doc.
- One commit per task, in Conventional Commits style, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not push or create a GitHub repo without asking the user first.

## Review Focus

1. **The real model ignores the reply format** (plan lines, route word, PASS/FAIL, arithmetic steps). The graph must finish on its documented fallback, not crash. Pinned by: p02 and p06 empty-plan tests, p08 garbage-route test, p11 garbage-grade test, p14 invalid-step test.
2. **The critic is never satisfied, or the agent never stops.** The loop must end at its cap constant. Pinned by: p03 `MAX_STEPS`, p04 `MAX_TRIALS`, p05 `MAX_ITERATIONS`, p11 `MAX_ATTEMPTS`, p12 `MAX_HOPS`.
3. **`USE_OPENROUTER=1` is set without a key.** This must raise a `RuntimeError` that names `OPENROUTER_API_KEY`, not an obscure OpenAI client error. Pinned by: `test_core`.
4. **Parallel branches finish in any order.** Results must be keyed or sorted, never positional. Pinned by: p06 results keyed by task id, p09 sorted summaries, p10 report in plan order, p13 `dict(verified)`.
5. **Fan-out over zero items or too many.** No hang and no runaway worker count. Pinned by: p09 empty-reviews test, p10 `MAX_SECTIONS` test.

## File map

```
agent-patterns-langgraph/
├── pyproject.toml, uv.lock, .gitignore, .env.example
├── README.md                              # quickstart, comparison table, decision guide
├── .claude/skills/add-pattern/SKILL.md    # daily recipe + module skeleton + doc template
├── agent_patterns/
│   ├── __init__.py                        # empty
│   ├── core.py                            # MockLLM, get_llm, run_demo, PATTERNS, mermaid_for, sync_docs
│   ├── p01_react/
│   │   ├── __init__.py                    # empty
│   │   ├── graph.py                       # build_graph, mock_llm, EXAMPLE_INPUT, __main__ demo
│   │   ├── simple.py                      # same pattern in plain Python: run(inputs, llm)
│   │   ├── test_graph.py                  # pattern tests
│   │   └── README.md                      # plain words + concept diagram + LangGraph diagram + when to use
│   └── p02_rewoo/ … p14_tree_of_thoughts/ # same 5 files each
└── tests/
    └── test_core.py, test_conventions.py, test_openrouter_smoke.py
```

Pattern tasks 2–15 each touch only their own folder plus one root README row, so they are independent of each other.

---

### Task 1: Scaffold, core, conventions test, `/add-pattern` skill

**Files:**
- Create: `pyproject.toml` (via uv), `.gitignore`, `.env.example`, `README.md`, `agent_patterns/__init__.py`, `agent_patterns/core.py`, `tests/test_core.py`, `tests/test_conventions.py`, `.claude/skills/add-pattern/SKILL.md`

**Interfaces:**
- Produces (`agent_patterns/core.py`):
  ```python
  OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
  DEFAULT_OPENROUTER_MODEL = "anthropic/claude-haiku-5.5"

  class MockLLM(BaseChatModel):
      responses: list[str | AIMessage] = []        # replayed in order, cycling
      by_keyword: dict[str, str | AIMessage] = {}  # checked first: first key that is a substring
                                                   # of the LAST message's content wins; doesn't advance i
      i: int = 0                                   # next index into responses
      calls: int = 0                               # every invoke
      prompts: list[str] = []                      # last-message content of every invoke, in order
      def bind_tools(self, tools, **kwargs) -> "MockLLM": ...   # returns self
      # implement _generate(...) -> ChatResult and _llm_type -> "mock".
      # No keyword match and no responses -> ValueError("MockLLM: no scripted reply for: <first 80 chars>")

  def get_llm(mock: Callable[[], BaseChatModel]) -> BaseChatModel
      # USE_OPENROUTER != "1" -> mock()
      # else ChatOpenAI(model=$OPENROUTER_MODEL or default, base_url=OPENROUTER_BASE_URL,
      #                 api_key=$OPENROUTER_API_KEY, temperature=0)
      # missing key -> RuntimeError("USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set")
  def run_demo(graph, inputs: dict) -> dict
      # graph.stream(inputs, stream_mode=["updates", "values"]); print "▶ <node>" + update
      # (each value repr truncated to 200 chars) per update; an update may be None (node returned {});
      # return the last "values" chunk
  PATTERNS: list[str]          # sorted names of dirs agent_patterns/p[0-9][0-9]_*/ that contain graph.py
                               # (Path.glob, no registry)
  def mermaid_for(name: str) -> str
      # import agent_patterns.<name>.graph; build_graph(mock_llm()).get_graph().draw_mermaid()
  def sync_docs() -> list[str]
      # for each name: replace the text between "<!-- mermaid:start -->" and "<!-- mermaid:end -->"
      # in agent_patterns/<name>/README.md with "\n```mermaid\n" + mermaid_for(name).strip() + "\n```\n";
      # return changed names
  # `python -m agent_patterns.core` -> print(sync_docs())
  ```

- [ ] **Step 1: Init the project**

```bash
cd /home/idanbe/Desktop/ai-projects/agent-patterns-langgraph
git init -b main
uv init --bare --python 3.12 --name agent-patterns-langgraph
uv add 'langgraph>=1.2' 'langchain-core>=1.7' 'langchain-openai>=1.7'
uv add --dev 'pytest>=9'
```
Add `[tool.pytest.ini_options]` with `pythonpath = ["."]` and `testpaths = ["tests", "agent_patterns"]` to `pyproject.toml`. Every pattern folder has an `__init__.py`, which lets the identical `test_graph.py` basenames coexist. `.gitignore` gets `.venv/`, `__pycache__/`, `.env` and `.pytest_cache/`. `.env.example` gets `OPENROUTER_API_KEY=` and `OPENROUTER_MODEL=anthropic/claude-haiku-5.5`.

- [ ] **Step 2: Write `tests/test_core.py`**

```python
import pytest
from typing import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_patterns.core import (MockLLM, get_llm, run_demo, PATTERNS,
                                 OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL)

def test_mock_replays_in_order_and_cycles():
    llm = MockLLM(responses=["a", "b"])
    assert [llm.invoke("x").content for _ in range(3)] == ["a", "b", "a"]
    assert llm.calls == 3 and llm.prompts == ["x", "x", "x"]

def test_mock_keyword_wins_and_does_not_advance_sequence():
    llm = MockLLM(responses=["seq"], by_keyword={"weather": "sunny"})
    assert llm.invoke("what is the weather?").content == "sunny"
    assert llm.invoke("other").content == "seq"

def test_mock_returns_tool_calls_and_bind_tools_is_noop():
    call = AIMessage("", tool_calls=[{"name": "add", "args": {"a": 1}, "id": "1"}])
    llm = MockLLM(responses=[call])
    assert llm.bind_tools([]) is llm
    assert llm.invoke("x").tool_calls[0]["name"] == "add"

def test_mock_without_script_raises_clear_error():
    with pytest.raises(ValueError, match="no scripted reply"):
        MockLLM().invoke("hello")

def test_get_llm_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("USE_OPENROUTER", raising=False)
    assert isinstance(get_llm(lambda: MockLLM(responses=["x"])), MockLLM)

def test_get_llm_uses_openrouter_when_enabled(monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    llm = get_llm(MockLLM)
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.model_name == DEFAULT_OPENROUTER_MODEL

def test_get_llm_openrouter_without_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_llm(MockLLM)

def test_run_demo_prints_nodes_and_returns_final_state(capsys):
    class S(TypedDict):
        n: int
    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc"); g.add_edge("inc", END)
    assert run_demo(g.compile(), {"n": 1}) == {"n": 2}
    assert "▶ inc" in capsys.readouterr().out

def test_patterns_discovered_and_numbered_consecutively():
    assert [int(n[1:3]) for n in PATTERNS] == list(range(1, len(PATTERNS) + 1))
```

- [ ] **Step 3: Run it.** `uv run pytest tests/test_core.py -q`. Expected: FAIL with `ModuleNotFoundError: agent_patterns.core`.

- [ ] **Step 4: Implement `agent_patterns/core.py`** per the Interfaces block. Create an empty `agent_patterns/__init__.py`.

- [ ] **Step 5: Run it.** `uv run pytest tests/test_core.py -q`. Expected: 9 passed.

- [ ] **Step 6: Write `tests/test_conventions.py`.** It is parametrized over `PATTERNS`, so it collects zero cases until Task 2.

```python
import importlib, re
from pathlib import Path
import pytest
from agent_patterns.core import PATTERNS, mermaid_for

ROOT = Path(__file__).resolve().parents[1]
MERMAID = re.compile(r"<!-- mermaid:start -->\n```mermaid\n(.*?)\n```\n<!-- mermaid:end -->", re.S)
LABEL = re.compile(r'[\[({]+"?([^"\])}]+?)"?[\])}]+')     # node text inside [..] (..) {..}
HEADINGS = ["## In plain words", "## How it works", "## The LangGraph graph", "## Plain Python vs LangGraph", "## Run it",
            "## When to use", "## When not to use", "## vs. closest pattern", "## Sources"]

def section(doc: str, heading: str) -> str:
    return doc.split(heading, 1)[1].split("\n## ", 1)[0]

@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_folder_follows_conventions(name):
    folder = ROOT / "agent_patterns" / name
    for f in ("__init__.py", "graph.py", "simple.py", "test_graph.py", "README.md"):
        assert (folder / f).exists(), f"missing {name}/{f}"
    mod = importlib.import_module(f"agent_patterns.{name}.graph")
    assert callable(mod.build_graph) and callable(mod.mock_llm) and isinstance(mod.EXAMPLE_INPUT, dict)
    doc = (folder / "README.md").read_text()
    assert doc.startswith(f"# {name[1:3]} · ")
    for h in HEADINGS:
        assert h in doc, f"missing {h}"
    concept = section(doc, "## How it works")
    assert "```mermaid" in concept, "add the hand-drawn concept diagram"
    for label in LABEL.findall(concept.split("```mermaid", 1)[1].split("```", 1)[0]):
        for line in label.split("<br/>"):
            assert len(line.strip()) <= 20, f"label too long for GitHub, split with <br/>: {line!r}"
    block = MERMAID.search(doc)
    assert block and block.group(1).strip() == mermaid_for(name).strip(), \
        "stale diagram: run `uv run python -m agent_patterns.core`"
    assert f"(agent_patterns/{name}/)" in (ROOT / "README.md").read_text(), "add a root README table row"
    mod.build_graph(mod.mock_llm()).invoke(mod.EXAMPLE_INPUT)   # example runs offline
    assert "langgraph" not in (folder / "simple.py").read_text(), "simple.py must be plain Python"
    importlib.import_module(f"agent_patterns.{name}.simple").run(mod.EXAMPLE_INPUT, mod.mock_llm())
```

- [ ] **Step 7: Write `README.md`** with these sections: a 2-line intro naming the source routine; `## Quickstart` (`uv sync`, `uv run pytest`, `uv run python -m agent_patterns.p01_react.graph`, `uv run python -m agent_patterns.p01_react.simple`); `## Real model (OpenRouter)` (`cp .env.example .env`, then `USE_OPENROUTER=1 uv run --env-file .env python -m agent_patterns.p01_react.graph`, then the `OPENROUTER_MODEL` override); `## Patterns` with this exact table header, which later tasks append rows to:

```markdown
| # | Pattern | Family | Idea | LLM calls* | Use when | Avoid when |
|---|---|---|---|---|---|---|
```
and the footnote `\* LLM calls in the repo's mock example — a rough cost/latency signal.` The last section is `## Adding a pattern`, which points to `/add-pattern <name>` and the skill file.

- [ ] **Step 8: Write `.claude/skills/add-pattern/SKILL.md`.** Frontmatter: `name: add-pattern`, `description: Use when adding a new agent design pattern to this repo (e.g. today's daily lesson) — creates the pattern folder (graph, test, README with diagrams) and the root README row`. The body holds:
  1. Recipe: (a) NN = `len(PATTERNS) + 1`, snake name. (b) Find the paper or official doc and verify every link with curl (200). (c) Create `agent_patterns/pNN_x/` with an empty `__init__.py`, then write `test_graph.py` first: assert final state values and `llm.calls`, one test per fallback and per loop cap, and `test_simple_matches_graph` (the parity template below). (d) Implement `graph.py` from the skeleton below, then `simple.py` from its skeleton: same helpers, plain control flow and obey this plan's Global Constraints: parallel work uses `Send` + reducers + `by_keyword` mocks, because branch order is nondeterministic. (e) Write the folder `README.md` from the template below, including the hand-drawn concept diagram. (f) Run `uv run python -m agent_patterns.core` to sync the diagram. (g) Add a root README table row and a node in the "Which pattern should I use?" flowchart. (h) `uv run pytest -q` is green and `uv run python -m agent_patterns.pNN_x.graph` shows the trace. (i) Commit `feat(pNN): add <Name> pattern`.
  2. Module skeleton (exact copy):
     ```python
     """NN · Name: one-line idea."""
     from typing import TypedDict
     from langchain_core.language_models import BaseChatModel
     from langgraph.graph import StateGraph, START, END
     from agent_patterns.core import MockLLM, get_llm, run_demo

     class State(TypedDict):
         ...

     def mock_llm() -> MockLLM:
         return MockLLM(responses=[...])

     def build_graph(llm: BaseChatModel | None = None):
         llm = get_llm(mock_llm) if llm is None else llm
         g = StateGraph(State)
         ...
         return g.compile()

     EXAMPLE_INPUT = {...}

     if __name__ == "__main__":
         run_demo(build_graph(), EXAMPLE_INPUT)
     ```
  2b. `simple.py` skeleton and parity test template (exact copy):
     ```python
     """NN · Name in plain Python: same prompts, tools and mock as graph.py; only the control flow differs."""
     from langchain_core.language_models import BaseChatModel
     from agent_patterns.core import get_llm
     from agent_patterns.pNN_x.graph import mock_llm, EXAMPLE_INPUT  # + prompts, tools, parsers, caps

     def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
         llm = get_llm(mock_llm) if llm is None else llm
         ...

     if __name__ == "__main__":
         print(run(EXAMPLE_INPUT))
     ```
     ```python
     def test_simple_matches_graph():
         from agent_patterns.pNN_x.simple import run   # local import: graph tests run before simple.py exists
         g_llm, s_llm = mock_llm(), mock_llm()
         g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
         s = run(EXAMPLE_INPUT, s_llm)
         # one assert per parity key; lists filled by parallel branches are compared sorted
         assert s_llm.calls == g_llm.calls
     ```
  3. Folder README template. Start with `# NN · Name`, then a `> one-sentence idea — Authors, Year ([arXiv id](url))` quote, then the 8 `HEADINGS` from `test_conventions.py`, in order:
     - `## In plain words`: 2–4 sentences with no jargon, plus one everyday analogy.
     - `## How it works`: a hand-drawn concept diagram (```` ```mermaid flowchart TD ````) that shows the *idea*, not the code. Use labelled decision edges (`-->|yes|`), at most 20 characters per label line, and `<br/>` for breaks. Below it, 3–6 numbered steps. Example (ReAct):
       ```mermaid
       flowchart TD
           Q["Question"] --> T["LLM thinks"]
           T --> D{"Tool needed?"}
           D -->|yes| A["Run tool"]
           A --> O["Observation"]
           O --> T
           D -->|no| F["Final answer"]
       ```
     - `## The LangGraph graph`: the empty marker pair `<!-- mermaid:start -->` / `<!-- mermaid:end -->`, which `python -m agent_patterns.core` fills. Then a `| Node | Role |` table and one line pointing to [graph.py](graph.py).
     - `## Plain Python vs LangGraph`: a link to [simple.py](simple.py), then 2–3 honest bullets on what the graph adds for this pattern (parallel branches, per-node trace, diagram, checkpoint/resume, human-in-the-loop) and when the plain loop is enough.
     - `## Run it`: the mock and OpenRouter commands for `graph` and `simple` (`python -m agent_patterns.pNN_x.graph` / `.simple`), plus `LLM calls in the mock example: **N**`.
     - `## When to use` / `## When not to use`: 2 bullets each. `## vs. closest pattern`: 1–2 lines. `## Sources`: curl-verified links.
     - Total length: about 80 lines or fewer.
  4. Next curriculum items, for reference: LATS, Supervisor multi-agent, Hierarchical teams, Handoffs/swarm, Multi-agent debate, Blackboard, Agentic RAG, CRAG, Self-RAG, Adaptive RAG, CodeAct, Program-of-Thought, Step-back, Human-in-the-loop, Agent memory, Guardrails.

- [ ] **Step 9: Verify.** `uv run pytest -q`. Expected: 9 passed, plus the conventions test skipped for its empty parameter set.

- [ ] **Step 10: Commit.** `git add -A && git commit -m "chore: scaffold repo, core mock/OpenRouter LLM, conventions and add-pattern skill"`

---

**Every pattern task (2–15) uses the same steps:**

- [ ] S1: Create `agent_patterns/pNN_x/__init__.py` (empty). Write `agent_patterns/pNN_x/test_graph.py` with the task's tests plus `test_simple_matches_graph` (skill template, using the task's **Parity** asserts).
- [ ] S2: `uv run pytest agent_patterns/pNN_x -q`. Expected: FAIL with `ModuleNotFoundError` (no `graph.py` yet).
- [ ] S3: Implement `agent_patterns/pNN_x/graph.py` from the skill skeleton plus the task's spec. Then `uv run pytest agent_patterns/pNN_x -q`. Expected: everything passes except `test_simple_matches_graph`, which fails because `simple.py` doesn't exist yet.
- [ ] S4: Implement `agent_patterns/pNN_x/simple.py` per the task's **simple.py** line. Then `uv run pytest agent_patterns/pNN_x -q`. Expected: all pass.
- [ ] S5: Write `agent_patterns/pNN_x/README.md` from the skill template, using the task's **Concept diagram**, **README facts** and **Plain vs graph** line. Run `uv run python -m agent_patterns.core` to fill the LangGraph diagram, then append the task's root README row.
- [ ] S6: `uv run pytest -q` all pass. `uv run python -m agent_patterns.pNN_x.graph` prints the `▶ node` trace, and `uv run python -m agent_patterns.pNN_x.simple` prints the same final answer.
- [ ] S7: `git add -A && git commit -m "feat(pNN): add <Name> pattern"`

Each task below gives its own checkboxes S1–S7.

---

### Task 2: 01 · ReAct

**Files:** `agent_patterns/p01_react/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `MessagesState`.
- Tools (`@tool`): `search(query: str) -> str` looks up `FACTS = {"population of france": "68 million"}` by first key contained in `query.lower()`, else `"no result"`. `multiply(a: float, b: float) -> float`.
- Nodes: `agent` = `llm.bind_tools(TOOLS).invoke([SystemMessage(SYSTEM)] + messages)`, and `tools` = `ToolNode(TOOLS)`.
- Edges: `START→agent`, `agent --tools_condition--> tools|END`, `tools→agent`.
- Mock `responses`: tool call `search{"query": "population of France"}` (id `c1`), tool call `multiply{"a": 68, "b": 2}` (id `c2`), then `"Twice France's population is about 136 million."`.
- `EXAMPLE_INPUT = {"messages": [HumanMessage("What is twice the population of France?")]}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Question` → `LLM thinks` → `{Tool needed?}` -yes→ `Run tool` → `Observation` → back to `LLM thinks`; -no→ `Final answer`.

**Test:**
```python
from langchain_core.messages import ToolMessage
from agent_patterns.p01_react.graph import build_graph, mock_llm, search, EXAMPLE_INPUT

def test_react_loops_thought_action_observation():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [m.content for m in out["messages"] if isinstance(m, ToolMessage)] == ["68 million", "136.0"]
    assert "136" in out["messages"][-1].content
    assert llm.calls == 3

def test_search_miss_returns_no_result():
    assert search.invoke({"query": "population of Mars"}) == "no result"
```

**simple.py:** `for _ in range(MAX_TURNS := 10)`: call `llm.bind_tools(TOOLS).invoke(messages)` and append the result. With no `tool_calls`, return `{"answer": ai.content, "observations": [...]}`. Otherwise run each call through `TOOLS_BY_NAME[name].invoke(args)` and append `ToolMessage(str(result), tool_call_id=id)`.

**Parity** (`test_simple_matches_graph`): `s["answer"] == g["messages"][-1].content`; `s["observations"] == [m.content for m in g["messages"] if isinstance(m, ToolMessage)]` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain loop is about 15 lines and fine for one agent. The graph adds a per-step trace, `ToolNode` error handling and `recursion_limit`, and adding human-in-the-loop later becomes one `interrupt`.

**README facts:** The LLM is called after every observation. The loop guard is `graph.invoke(x, {"recursion_limit": 10})`. **vs ReWOO:** ReAct re-plans after each tool result, which adapts better but costs more calls and tokens.

**README row:** `| 01 | [ReAct](agent_patterns/p01_react/) | Tool agent | Think → act → observe loop, one LLM call per step | 3 | Steps unknown upfront; must react to tool results | Fixed workflows; tight latency/cost budgets |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 3: 02 · ReWOO

**Files:** `agent_patterns/p02_rewoo/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `task: str`, `plan: str`, `steps: list[tuple[str, str, str]]` (var, tool, arg), `results: dict[str, str]`, `result: str`.
- `STEP_RE = re.compile(r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]")`; `steps = [(v, t, a) for _, v, t, a in STEP_RE.findall(plan)]`.
- Nodes:
  - `plan`: one LLM call, returns `plan` and `steps`. The prompt documents the tools `Search[query]` and `LLM[prompt]` and the `#En` variables.
  - `work`: no LLM reasoning. Calls the module-level `run_steps(steps, llm) -> dict[str, str]`, which runs `steps` in order. Before each call it replaces earlier `#En` in the arg, longest key first. `Search` uses `FACTS = {"2018 fifa world cup winner": "France", "capital of france": "Paris"}` (first key in `arg.lower()`, else `"unknown"`). `LLM` uses `llm.invoke(arg).content`. Any other tool gives `"unknown tool"`.
  - `solve`: one LLM call with the plan and evidence, returns `result`.
- Edges: `START→plan→work→solve→END`.
- `MOCK_PLAN = "Plan: Find the 2018 World Cup winner.\n#E1 = Search[2018 FIFA World Cup winner]\nPlan: Find that country's capital.\n#E2 = Search[capital of #E1]"`. Mock `responses = [MOCK_PLAN, "France won the 2018 World Cup; its capital is Paris."]`.
- `EXAMPLE_INPUT = {"task": "What is the capital of the country that won the 2018 FIFA World Cup?"}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Planner` (one call) → `Plan with #E slots` → `Worker runs tools` (subgraph showing `Search 1` → `Search 2` in order) → `Solver` → `Answer`. Note under it: no LLM between tools.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p02_rewoo.graph import build_graph, mock_llm, EXAMPLE_INPUT

def test_rewoo_plans_once_runs_tools_then_solves():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["steps"] == [("#E1", "Search", "2018 FIFA World Cup winner"), ("#E2", "Search", "capital of #E1")]
    assert out["results"] == {"#E1": "France", "#E2": "Paris"}
    assert "Paris" in out["result"]
    assert llm.calls == 2          # planner + solver only

def test_unparseable_plan_goes_straight_to_solver():
    llm = MockLLM(responses=["I will just answer.", "Paris"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["results"] == {} and out["result"] == "Paris" and llm.calls == 2
```

**simple.py:** three straight calls: plan → `run_steps(steps, llm)` (module-level helper in graph.py, also used by the `work` node) → solve.

**Parity** (`test_simple_matches_graph`): `results`, `result` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The flow is linear, so plain Python is just as clear. The graph only adds a trace and persistence.

**README facts:** The LLM call count stays at 2 no matter how many tools run. **vs ReAct:** ReWOO plans with variable placeholders and never observes between steps, so it cannot adapt if step 1 returns a surprise.

**README row:** `| 02 | [ReWOO](agent_patterns/p02_rewoo/) | Tool agent | Plan every tool call once with #E variables, run them, then solve | 2 | Predictable multi-tool lookups; fewer tokens than ReAct | Later steps depend on surprises in earlier results |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 4: 03 · Plan-and-Execute

**Files:** `agent_patterns/p03_plan_execute/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `input: str`, `plan: list[str]`, `past_steps: Annotated[list[tuple[str, str]], operator.add]`, `response: str`.
- `parse_plan(text: str) -> list[str]` = `re.findall(r"^\s*\d+[.)]\s*(.+)$", text, re.M)`. `MAX_STEPS = 6`.
- Nodes:
  - `planner`: `plan = parse_plan(reply) or [input]`.
  - `executor`: one LLM call on `plan[0]` with past steps as context, appends `(plan[0], reply)`.
  - `replanner`: first, if `len(past_steps) >= MAX_STEPS`, return `response = f"Stopped after {MAX_STEPS} steps."` **without** calling the LLM. Otherwise call the LLM. If the reply contains `FINAL:`, set `response` to the text after it, stripped. Otherwise set `plan = parse_plan(reply)`, and if that is empty set `response = reply`.
- Edges: `START→planner→executor→replanner`; `replanner → END if response else executor`.
- Mock `responses`: `"1. Look up the hotel price per night in Paris\n2. Add a daily food budget of 60 EUR"`, `"Hotel: 150 EUR per night"`, `"1. Add a daily food budget of 60 EUR"`, `"150 + 60 = 210 EUR per day"`, `"FINAL: Budget about 210 EUR per day (hotel 150 + food 60)."`
- `EXAMPLE_INPUT = {"input": "Estimate a one-day budget in Paris: hotel plus food.", "past_steps": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Planner` → `Step list` → `Executor: 1 step` → `Replanner` → `{Done?}` -no→ `Executor: 1 step`; -yes→ `Answer`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p03_plan_execute.graph import build_graph, mock_llm, parse_plan, EXAMPLE_INPUT, MAX_STEPS

def test_plan_execute_replans_after_each_step():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert [s for s, _ in out["past_steps"]] == ["Look up the hotel price per night in Paris",
                                                 "Add a daily food budget of 60 EUR"]
    assert out["response"].startswith("Budget about 210 EUR")
    assert llm.calls == 5

def test_replanner_that_never_finishes_is_capped():
    llm = MockLLM(responses=["1. step"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert len(out["past_steps"]) == MAX_STEPS == 6
    assert out["response"].startswith("Stopped")
    assert llm.calls == 12         # planner + 6 executor + 5 replanner

def test_parse_plan_tolerates_noise():
    assert parse_plan("1. a\n2) b\nnoise") == ["a", "b"]
```

**simple.py:** a `while` loop with the same `MAX_STEPS` cap and `FINAL:` check.

**Parity** (`test_simple_matches_graph`): `response`, `past_steps` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The graph makes the replan loop visible, and with a checkpointer a long task can resume after a crash.

**README facts:** In production the executor is usually a ReAct sub-agent running on a cheaper model. **vs ReWOO:** both plan upfront, but Plan-and-Execute revisits the plan after every step.

**README row:** `| 03 | [Plan-and-Execute](agent_patterns/p03_plan_execute/) | Tool agent | Planner writes steps, executor does one, replanner adjusts | 5 | Long multi-step tasks; strong planner + cheap executor | Short tasks (planning overhead); hard real-time |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 5: 04 · Reflexion

**Files:** `agent_patterns/p04_reflexion/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `task: str`, `attempt: str`, `feedback: list[str]`, `reflections: Annotated[list[str], operator.add]`, `trials: int`, `passed: bool`. `MAX_TRIALS = 3`.
- `check_slogan(text: str) -> list[str]` is the external evaluator (no LLM). It returns failures:
  - `f"too long ({n} words, max 8)"` if `len(text.split()) > 8`
  - `"must mention 'planet'"` if `"planet"` is not in `text.lower()`
  - `"no exclamation marks"` if the text contains `"!"`
- Nodes:
  - `actor`: one LLM call. The prompt includes the task and every reflection as `"Lessons from earlier attempts:\n- …"`. Returns `attempt` and `trials + 1`.
  - `evaluate`: `feedback = check_slogan(attempt)`, `passed = not feedback`.
  - `reflect`: one LLM call given the attempt and feedback, appends to `reflections`.
- Edges: `START→actor→evaluate`; `evaluate → END if passed or trials >= MAX_TRIALS else reflect`; `reflect→actor`.
- Mock `responses`: `"Stay hydrated and save the planet every single day!"`, `"I wrote 9 words and used '!'. Next time: at most 8 words, no exclamation marks."`, `"Refill daily, protect the planet."`
- `EXAMPLE_INPUT = {"task": "Write a slogan for a reusable water bottle. Max 8 words, mention 'planet', no exclamation marks.", "trials": 0, "reflections": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Actor: attempt` → `Evaluator<br/>(external check)` → `{Passed?}` -yes→ `Done`; -no→ `Self-reflect` → `Memory: lessons` → back to `Actor: attempt`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p04_reflexion.graph import build_graph, mock_llm, check_slogan, EXAMPLE_INPUT, MAX_TRIALS

def test_reflexion_learns_from_external_feedback():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["passed"] and out["trials"] == 2
    assert out["attempt"] == "Refill daily, protect the planet."
    assert len(out["reflections"]) == 1
    assert out["reflections"][0] in llm.prompts[2]     # 2nd attempt saw the lesson
    assert llm.calls == 3

def test_reflexion_stops_after_max_trials():
    llm = MockLLM(responses=["Buy now!"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert not out["passed"] and out["trials"] == MAX_TRIALS == 3
    assert llm.calls == 5          # 3 attempts + 2 reflections

def test_check_slogan():
    assert check_slogan("Refill daily, protect the planet.") == []
    assert len(check_slogan("Stay hydrated and save the planet every single day!")) == 2
```

**simple.py:** a `for trial in range(1, MAX_TRIALS + 1)` loop: attempt → `check_slogan` → reflect, but only if `trial < MAX_TRIALS`.

**Parity** (`test_simple_matches_graph`): `attempt`, `passed`, `trials`, `reflections` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain loop is shorter. The graph pays off when trials are slow or costly and you want checkpoints, or a human approval between trials.

**README facts:** The paper's evaluators are unit tests, environment rewards and heuristics, and reflections act as episodic memory. **vs Self-Refine:** Reflexion needs an *external* signal and keeps lessons across trials. Self-Refine is self-judged within one task.

**README row:** `| 04 | [Reflexion](agent_patterns/p04_reflexion/) | Self-correction | Retry with verbal lessons learned from an external evaluator | 3 | You have a ground-truth check (tests, rules, env reward) | No reliable evaluator; one-shot tasks |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 6: 05 · Self-Refine

**Files:** `agent_patterns/p05_self_refine/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `task: str`, `draft: str`, `feedback: str`, `history: Annotated[list[str], operator.add]`, `iterations: int`. `MAX_ITERATIONS = 3`.
- Nodes:
  - `generate`: produces `draft`.
  - `feedback`: the same LLM critiques its own draft. The prompt says to reply `NO_ISSUES` if the draft is good. Returns `iterations + 1`.
  - `refine`: rewrites the draft using the feedback, and appends the old draft to `history`.
- Edges: `START→generate→feedback`; `feedback → END if "NO_ISSUES" in feedback or iterations >= MAX_ITERATIONS else refine`; `refine→feedback`.
- Mock `responses`: `"fixed bug"`, `"Too vague: say what was fixed and where."`, `"fix(profile): handle users without an email address"`, `"NO_ISSUES"`.
- `EXAMPLE_INPUT = {"task": "Write a git commit message for: fixed a crash when a user has no email", "iterations": 0, "history": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Draft` → `Self-feedback` → `{Any issues?}` -no→ `Done`; -yes→ `Refine draft` → back to `Self-feedback`. All three boxes carry the same `same LLM` style class.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p05_self_refine.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_ITERATIONS

def test_self_refine_iterates_until_no_issues():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["draft"] == "fix(profile): handle users without an email address"
    assert out["history"] == ["fixed bug"]
    assert out["iterations"] == 2 and llm.calls == 4

def test_self_refine_stops_at_max_iterations():
    llm = MockLLM(responses=["meh"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["iterations"] == MAX_ITERATIONS == 3
    assert llm.calls == 6          # generate + 3 feedback + 2 refine
```

**simple.py:** a `while` loop with the same `MAX_ITERATIONS` cap and `NO_ISSUES` stop word.

**Parity** (`test_simple_matches_graph`): `draft`, `history`, `iterations` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain loop wins on clarity. The graph adds a trace of each critique round.

**README facts:** One model plays three roles. It cannot catch errors it doesn't know are errors. **vs Evaluator-optimizer:** that pattern uses a separate judge role with explicit PASS/FAIL criteria, which can be a different model.

**README row:** `| 05 | [Self-Refine](agent_patterns/p05_self_refine/) | Self-correction | Same model drafts, critiques itself, refines until no issues | 4 | Polishing writing/code style; no external checker | Factual errors the model can't see; cost-sensitive |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 7: 06 · LLMCompiler

**Files:** `agent_patterns/p06_llm_compiler/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `question: str`, `tasks: dict[int, dict]` (`{"tool", "arg", "deps": set[int]}`), `results: Annotated[dict[int, str], operator.or_]`, `waves: Annotated[list[list[int]], operator.add]`, `answer: str`.
- `parse_tasks(text) -> dict[int, dict]`: `re.finditer(r"^\s*(\d+)\.\s*(\w+)\((.*)\)\s*$", text, re.M)`. `arg` is stripped of whitespace and surrounding quotes. `deps = {int(d) for d in re.findall(r"\$(\d+)", arg)}`.
- `ready_tasks(tasks, results) -> list[int]`: sorted ids not in `results` whose `deps <= results.keys()`.
- Tools:
  - `search(q)`: `FACTS = {"population of japan": "125", "population of germany": "84"}`, else `"unknown"`.
  - `subtract(arg)`: splits `"a, b"` and returns `f"{a - b:g}"`, or `"error"` on bad input.
- Nodes and edges (the barrier scheduler, verified on langgraph 1.2.20):
  - `plan`: one LLM call, returns `tasks`.
  - `schedule`: returns `{"waves": [ready]}` if any are ready, else `{}`.
  - Conditional edge `dispatch(state)`: `[Send("run_task", {"id": t, "task": tasks[t], "results": results}) for t in ready]` or `"join"`.
  - `execute(task: dict, results: dict[int, str]) -> str` (module-level): replaces `$K` with `results[K]`, longest id first so `$10` is not hit by `$1`, then runs the tool. The `run_task` node returns `{"results": {id: execute(task, results)}}`.
  - `join`: one LLM call, returns `answer`.
  - `START→plan→schedule`, `schedule --dispatch--> ["run_task", "join"]`, `run_task→schedule`, `join→END`. Tasks whose deps never resolve are skipped, and the graph goes to `join`.
- `MOCK_PLAN = '1. search("population of Japan")\n2. search("population of Germany")\n3. subtract($1, $2)'`. Mock `responses = [MOCK_PLAN, "Japan has the larger population, by about 41 million."]`.
- `EXAMPLE_INPUT = {"question": "Which has more people, Japan or Germany, and by how many millions?", "results": {}, "waves": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Question` → `Planner:<br/>task graph` → `Scheduler` → `Task 1` and `Task 2` in parallel → `Task 3<br/>(needs 1 and 2)` → `Joiner` → `Answer`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p06_llm_compiler.graph import build_graph, mock_llm, parse_tasks, MOCK_PLAN, EXAMPLE_INPUT

def test_independent_tasks_run_in_parallel_waves():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["waves"] == [[1, 2], [3]]
    assert out["results"] == {1: "125", 2: "84", 3: "41"}
    assert "41" in out["answer"] and llm.calls == 2

def test_empty_plan_goes_straight_to_joiner():
    llm = MockLLM(responses=["no tools needed", "answer"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["waves"] == [] and out["answer"] == "answer"

def test_parse_tasks_extracts_dependencies():
    assert parse_tasks(MOCK_PLAN)[3] == {"tool": "subtract", "arg": "$1, $2", "deps": {1, 2}}
```

**simple.py:** `while ready := ready_tasks(tasks, results)`: run that wave with `ThreadPoolExecutor().map(lambda t: execute(tasks[t], results), ready)`, merge into `results` and append the wave. Then join.

**Parity** (`test_simple_matches_graph`): `waves`, `results`, `answer` equal, and the same `llm.calls`.

**Plain vs graph** (README section): This pattern gains the most. `Send` fan-out plus reducers replace manual thread pools, dict merging and wave bookkeeping, and the trace shows each parallel task.

**README facts:** The paper also lets the joiner trigger a replan, which this repo skips (`ponytail:` comment). **vs ReWOO:** both plan upfront, but LLMCompiler plans a DAG and runs independent calls in parallel.

**README row:** `| 06 | [LLMCompiler](agent_patterns/p06_llm_compiler/) | Tool agent | Planner emits a task DAG; independent tool calls run in parallel | 2 | Many independent tool calls; latency matters | Strictly sequential tasks; side-effecting tools that need ordering |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 8: 07 · Prompt chaining

**Files:** `agent_patterns/p07_prompt_chaining/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `topic: str`, `outline: str`, `draft: str`, `translation: str`, `error: str`.
- Nodes:
  - `outline`: one LLM call. The prompt asks for exactly 3 bullets, each line starting with `- `.
  - `gate`: code, no LLM. Counts lines starting with `- `. If fewer than 3, returns `error = f"outline has {n} bullet(s), need 3"`.
  - `write`: one LLM call, produces a paragraph.
  - `translate`: one LLM call, Hebrew.
- Edges: `START→outline→gate`; `gate → END if error else write`; `write→translate→END`.
- Mock `responses`: `"- Keeps drinks cold 24h\n- Made from recycled steel\n- Lifetime warranty"`, `"Meet the bottle that keeps drinks cold for 24 hours, is made from recycled steel and comes with a lifetime warranty."`, `"הכירו את הבקבוק ששומר על משקאות קרים 24 שעות, עשוי מפלדה ממוחזרת ומגיע עם אחריות לכל החיים."`
- `EXAMPLE_INPUT = {"topic": "product blurb for an insulated steel water bottle"}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Topic` → `LLM 1: outline` → `{3 bullets?}` -no→ `Stop with error`; -yes→ `LLM 2: write` → `LLM 3: translate` → `Result`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p07_prompt_chaining.graph import build_graph, mock_llm, EXAMPLE_INPUT

def test_chain_runs_fixed_steps_in_order():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["outline"].count("- ") == 3
    assert "recycled steel" in out["draft"]
    assert "בקבוק" in out["translation"]
    assert llm.calls == 3

def test_gate_stops_chain_on_bad_outline():
    llm = MockLLM(responses=["- only one point"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert "outline" in out["error"] and "draft" not in out and llm.calls == 1
```

**simple.py:** four straight steps with one `if` for the gate.

**Parity** (`test_simple_matches_graph`): `outline`, `draft`, `translation` equal, and the same `llm.calls`.

**Plain vs graph** (README section): Plain Python is clearer for a fixed chain. The graph adds a per-step trace and retry or resume of a failed step.

**README facts:** The gate is plain code, so it costs nothing and is deterministic. Not every node needs an LLM. **vs Plan-and-Execute:** here the steps are fixed in code, where Plan-and-Execute has the LLM decide them.

**README row:** `| 07 | [Prompt chaining](agent_patterns/p07_prompt_chaining/) | Workflow | Fixed sequence of LLM calls with code gates between them | 3 | Task splits into known steps; want checkable intermediates | Steps vary per input |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 9: 08 · Routing

**Files:** `agent_patterns/p08_routing/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `ticket: str`, `route: str`, `answer: str`. `ROUTES = ("billing", "technical", "general")`.
- Nodes:
  - `classify`: one LLM call that replies with one word. `route = reply.strip().lower().strip(".")`, falling back to `"general"` if the result is not in `ROUTES`.
  - One handler node per route, built in a loop. Each makes one LLM call with its own `HANDLER_PROMPTS[route]`.
- Edges: `START→classify`; `add_conditional_edges("classify", lambda s: s["route"], {r: r for r in ROUTES})`; each handler goes `→END`.
- Mock `responses`: `"billing"`, `"You were charged twice; we've refunded the duplicate payment."`
- `EXAMPLE_INPUT = {"ticket": "I was charged twice for my subscription this month."}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Ticket` → `Classifier LLM` → `{Which kind?}` -billing→ `Billing handler`; -technical→ `Tech handler`; -other→ `General handler`; all three → `Answer`.

**Test:**
```python
import pytest
from agent_patterns.core import MockLLM
from agent_patterns.p08_routing.graph import build_graph, mock_llm, EXAMPLE_INPUT

def test_example_routes_to_billing():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["route"] == "billing" and "refunded" in out["answer"] and llm.calls == 2

@pytest.mark.parametrize("label,route", [("billing", "billing"), ("Technical.", "technical"), ("banana", "general")])
def test_router_normalizes_label_and_falls_back(label, route):
    llm = MockLLM(responses=[label, f"{route} answer"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["route"] == route and out["answer"] == f"{route} answer"
```

**simple.py:** `HANDLERS[route]`, a dict of route → handler function, against conditional edges.

**Parity** (`test_simple_matches_graph`): `route`, `answer` equal, and the same `llm.calls`.

**Plain vs graph** (README section): About equal in size. The graph's diagram shows every route at a glance.

**README facts:** Each route can use a different model, for example a cheap one for "general". **vs Orchestrator-workers:** a router picks ONE path, while an orchestrator fans out to many.

**README row:** `| 08 | [Routing](agent_patterns/p08_routing/) | Workflow | Classify the input, send it to one specialized handler | 2 | Distinct input categories needing different prompts/models | Categories overlap; one prompt handles all well |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 10: 09 · Parallelization (map-reduce)

**Files:** `agent_patterns/p09_map_reduce/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `reviews: list[str]`, `summaries: Annotated[list[str], operator.add]`, `summary: str`.
- Fan-out `fan_out(state)` returns `[Send("summarize", {"review": r}) for r in reviews]`, or `"combine"` when `reviews` is empty.
- Nodes:
  - `summarize`: one LLM call per review with the prompt `"Summarize this review in 5 words: {review}"`.
  - `combine`: if `summaries` is empty, returns `summary = "No reviews."` with no LLM call. Otherwise makes one LLM call with the prompt `"Combine these review summaries into one verdict: " + "; ".join(sorted(summaries))`.
- Edges: `add_conditional_edges(START, fan_out, ["summarize", "combine"])`, `summarize→combine→END`.
- Mock: `by_keyword` only, with `"Combine"` first: `{"Combine": "Great battery and value, but the screen is too dim.", "battery": "Battery life is excellent", "screen": "Screen is too dim", "price": "Good value for the price"}`.
- `EXAMPLE_INPUT = {"reviews": ["The battery lasts two days, love it.", "The screen is hard to read outdoors.", "For this price it is a steal."], "summaries": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Reviews` → `Split` → `Summarize 1`, `Summarize 2` and `Summarize 3` (parallel) → `Combine` → `Verdict`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p09_map_reduce.graph import build_graph, mock_llm, EXAMPLE_INPUT

def test_map_in_parallel_then_reduce():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert sorted(out["summaries"]) == ["Battery life is excellent", "Good value for the price", "Screen is too dim"]
    assert out["summary"].startswith("Great battery") and llm.calls == 4

def test_fan_out_scales_with_input():
    llm = MockLLM(by_keyword={"Combine": "ok", "x": "s"})
    out = build_graph(llm).invoke({"reviews": ["x"] * 5, "summaries": []})
    assert len(out["summaries"]) == 5 and llm.calls == 6

def test_no_reviews_skips_llm():
    llm = MockLLM()
    out = build_graph(llm).invoke({"reviews": [], "summaries": []})
    assert out["summary"] == "No reviews." and llm.calls == 0
```

**simple.py:** `ThreadPoolExecutor().map(summarize_one, reviews)`, then combine. Same empty-input shortcut.

**Parity** (`test_simple_matches_graph`): `sorted(s["summaries"]) == sorted(g["summaries"])`, `summary` equal, and the same `llm.calls`.

**Plain vs graph** (README section): Similar size. `Send` plus a reducer merges branch results for you, and each branch streams on its own.

**README facts:** Mention the voting variant: run the same prompt N times and take the majority. The mock uses `by_keyword` because parallel branch order is nondeterministic. **vs Orchestrator-workers:** here the split is fixed by code (one branch per input), where the orchestrator's LLM chooses the split.

**README row:** `| 09 | [Parallelization (map-reduce)](agent_patterns/p09_map_reduce/) | Workflow | Same LLM step on many inputs in parallel, then combine | 4 | Many independent chunks/docs; voting for confidence | Items depend on each other; strict rate limits |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 11: 10 · Orchestrator-workers

**Files:** `agent_patterns/p10_orchestrator_workers/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `task: str`, `sections: list[str]`, `drafts: Annotated[list[tuple[str, str]], operator.add]`, `report: str`. `MAX_SECTIONS = 5`.
- Nodes:
  - `orchestrator`: one LLM call. Its prompt asks for section titles one per line and **must not contain the substring `"Section:"`**. `sections = [l.strip("-• ").strip() for l in reply.splitlines() if l.strip()][:MAX_SECTIONS] or ["Overview"]`.
  - Conditional `assign` sends `Send("worker", {"task": task, "section": s})` for each section.
  - `worker`: one LLM call per section. The prompt ends with `"\nSection: {section}\n"`. Returns `{"drafts": [(section, reply)]}`.
  - `synthesize`: code, no LLM. Calls the module-level `assemble(sections: list[str], drafts: list[tuple[str, str]]) -> str` = `"\n\n".join(f"## {s}\n{dict(drafts)[s]}" for s in sections)`.
- Edges: `START→orchestrator`, `add_conditional_edges("orchestrator", assign, ["worker"])`, `worker→synthesize→END`.
- Mock: `MockLLM(responses=["Installation\nUsage\nLicense"], by_keyword={"Section: Installation": "Run `pip install todo-cli`.", "Section: Usage": 'Run `todo add "buy milk"`.', "Section: License": "MIT."})`
- `EXAMPLE_INPUT = {"task": "Write a short README for todo-cli, a command-line todo app.", "drafts": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Orchestrator LLM` -decides→ `Worker A`, `Worker B` and `Worker C` (parallel) → `Synthesizer` → `Report`. Note: the number of workers is chosen at runtime.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p10_orchestrator_workers.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_SECTIONS

def test_orchestrator_plans_sections_workers_fill_them():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["sections"] == ["Installation", "Usage", "License"]
    assert out["report"].startswith("## Installation\nRun `pip install todo-cli`.")
    assert out["report"].index("## Usage") < out["report"].index("## License")
    assert llm.calls == 4

def test_worker_count_is_capped():
    llm = MockLLM(responses=["\n".join(f"S{i}" for i in range(10))], by_keyword={"Section:": "text"})
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert len(out["sections"]) == MAX_SECTIONS == 5 and llm.calls == 6
```

**simple.py:** plan sections → `ThreadPoolExecutor().map(write_section, sections)` → `assemble(sections, drafts)` (shared helper).

**Parity** (`test_simple_matches_graph`): `sections`, `report` equal, and the same `llm.calls`.

**Plain vs graph** (README section): Like map-reduce, but the number of workers is decided at runtime, which `Send` handles natively.

**README facts:** The report is assembled in plan order no matter which worker finishes first. **vs Map-reduce:** the subtasks are decided by the LLM at runtime, not by code.

**README row:** `| 10 | [Orchestrator-workers](agent_patterns/p10_orchestrator_workers/) | Workflow | LLM decides subtasks at runtime; workers run in parallel; results merged | 4 | Subtasks not known until you see the input | Fixed splits (use map-reduce); tiny tasks |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 12: 11 · Evaluator-optimizer

**Files:** `agent_patterns/p11_evaluator_optimizer/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `task: str`, `draft: str`, `grade: str` (`"pass"` | `"fail"`), `feedback: str`, `attempts: int`. `MAX_ATTEMPTS = 3`.
- Nodes:
  - `generate`: the prompt includes the task and, when set, `"Feedback on the last attempt: {feedback}"`. Returns `attempts + 1`.
  - `evaluate`: the judge prompt lists the criteria (correct, warm and friendly tone, natural Hebrew) and asks for `PASS` or `FAIL: <feedback>`. Uses the module-level `parse_grade(reply: str) -> tuple[str, str]`, which returns `(grade, feedback)`: `grade = "pass" if reply.strip().upper().startswith("PASS") else "fail"`, and `feedback` is the text after `FAIL:`, or else the whole reply.
- Edges: `START→generate→evaluate`; `evaluate → END if grade == "pass" or attempts >= MAX_ATTEMPTS else generate`.
- Mock `responses`: `"הזמנתך נשלחה."`, `"FAIL: Correct but cold and formal; make it warm and friendly."`, `"איזה כיף, ההזמנה שלך כבר בדרך אליך!"`, `"PASS"`.
- `EXAMPLE_INPUT = {"task": "Translate to Hebrew in a warm, friendly tone: 'Your order has shipped.'", "attempts": 0, "feedback": ""}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Task` → `Generator` → `Evaluator<br/>(rubric)` → `{PASS?}` -yes→ `Done`; -no + feedback→ back to `Generator`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p11_evaluator_optimizer.graph import build_graph, mock_llm, EXAMPLE_INPUT, MAX_ATTEMPTS

def test_loops_until_judge_passes():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["grade"] == "pass" and out["attempts"] == 2
    assert out["draft"] == "איזה כיף, ההזמנה שלך כבר בדרך אליך!"
    assert "cold and formal" in llm.prompts[2]       # generator saw the judge's feedback
    assert llm.calls == 4

def test_unparseable_grade_is_fail_and_loop_is_capped():
    llm = MockLLM(responses=["draft", "looks fine I guess"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["grade"] == "fail" and out["attempts"] == MAX_ATTEMPTS == 3 and llm.calls == 6
```

**simple.py:** a `while` loop with the same `MAX_ATTEMPTS` cap and the same `parse_grade` helper.

**Parity** (`test_simple_matches_graph`): `draft`, `grade`, `attempts` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain loop is shorter. In the graph, a human can approve or override the judge between rounds.

**README facts:** In production the judge is often a different, stronger model. Pass a second `BaseChatModel` and you have it (not built here: `ponytail:` comment). **vs Self-Refine:** this pattern has a separate judge with a rubric and a hard PASS/FAIL.

**README row:** `| 11 | [Evaluator-optimizer](agent_patterns/p11_evaluator_optimizer/) | Workflow | Generator + separate judge loop until PASS | 4 | Clear quality rubric; iterating measurably helps | No articulable criteria; first draft usually fine |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 13: 12 · Self-Ask

**Files:** `agent_patterns/p12_self_ask/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `question: str`, `scratchpad: str`, `pending: str`, `hops: int`, `answer: str`. `MAX_HOPS = 4`.
- `search(q: str) -> str`: `FACTS = {"eiffel tower": "1889", "president of the united states in 1889": "Benjamin Harrison"}`, first key in `q.lower()`, else `"unknown"`.
- Nodes:
  - `ask`: first, if `hops >= MAX_HOPS`, return `answer = "unknown (follow-up limit reached)"` with no LLM call. Otherwise one LLM call with a prompt that explains the `Follow up:` / `Intermediate answer:` / `So the final answer is:` format, then `Question: …`, `Are follow up questions needed here: Yes.` and the scratchpad, and asks for the NEXT line only. If the reply contains `So the final answer is:`, set `answer` to the text after it. Else if it contains `Follow up:`, append the line to the scratchpad and set `pending` and `hops + 1`. Else set `answer = reply`.
  - `search`: appends `"\nIntermediate answer: {search(pending)}"` to the scratchpad.
- Edges: `START→ask`; `ask → END if answer else search`; `search→ask`.
- Mock `responses`: `"Follow up: When was the Eiffel Tower completed?"`, `"Follow up: Who was president of the United States in 1889?"`, `"So the final answer is: Benjamin Harrison"`.
- `EXAMPLE_INPUT = {"question": "Who was president of the United States when the Eiffel Tower was completed?", "scratchpad": "", "hops": 0}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Question` → `LLM: next line` → `{Follow-up?}` -yes→ `Search` → `Intermediate answer` → back to `LLM: next line`; -final→ `Answer`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p12_self_ask.graph import build_graph, mock_llm, search, EXAMPLE_INPUT, MAX_HOPS

def test_self_ask_decomposes_into_follow_ups():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["answer"] == "Benjamin Harrison"
    assert out["scratchpad"].count("Intermediate answer:") == 2
    assert "Intermediate answer: 1889" in out["scratchpad"]
    assert llm.calls == 3

def test_endless_follow_ups_are_capped():
    llm = MockLLM(responses=["Follow up: again?"])
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["answer"].startswith("unknown") and llm.calls == MAX_HOPS == 4

def test_search_miss_is_unknown():
    assert search("What is the GDP of Atlantis?") == "unknown"
```

**simple.py:** a `while` loop that appends to the scratchpad, with the same `MAX_HOPS` cap.

**Parity** (`test_simple_matches_graph`): `answer`, `scratchpad` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain loop is shorter. The graph adds a trace of each hop.

**README facts:** The paper names the "compositionality gap": a model can know both facts and still fail to compose them. **vs ReAct:** Self-Ask is ReAct narrowed to a single search tool, with an explicit sub-question format. It is the better fit for multi-hop QA.

**README row:** `| 12 | [Self-Ask](agent_patterns/p12_self_ask/) | Tool agent | Model asks itself follow-up questions; search answers each | 3 | Multi-hop factual questions | Single-hop questions; non-factual tasks |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 14: 13 · Chain-of-Verification

**Files:** `agent_patterns/p13_chain_of_verification/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec:**
- State: `question: str`, `baseline: str`, `checks: list[str]`, `verified: Annotated[list[tuple[str, str]], operator.add]`, `final: str`. `MAX_CHECKS = 5`.
- Prompt openings (these are mock keys, so the exact wording matters):
  - `BASELINE` = `"Answer the question: {question}"`
  - `PLAN` = `"Write one yes/no verification question per claim in this answer, one per line.\nQuestion: {question}\nAnswer: {baseline}"`
  - `VERIFY` = `"Answer briefly and factually: {check}"`. This is **factored**: the baseline is never included.
  - `FINAL` = `"Revise the draft answer using the verified facts; drop anything they contradict.\n…"`
- Nodes:
  - `baseline`: one LLM call.
  - `plan_checks`: `checks = [l.strip() for l in reply.splitlines() if l.strip().endswith("?")][:MAX_CHECKS]`.
  - Conditional `fan_out` sends `Send("verify", {"check": q})` for each check, or `"final"` if there are none.
  - `verify`: one LLM call, returns `{"verified": [(check, reply)]}`.
  - `final`: one LLM call.
- Edges: `START→baseline→plan_checks --fan_out--> ["verify", "final"]`, `verify→final→END`.
- Mock: `by_keyword` only, in this order: `{"Revise": "Go and Dart. (Python was not created at Google.)", "verification question": "Was Go created at Google?\nWas Dart created at Google?\nWas Python created at Google?", "Was Go created": "Yes, at Google in 2009.", "Was Dart created": "Yes, at Google in 2011.", "Was Python created": "No, Guido van Rossum created it at CWI.", "List three": "Go, Dart and Python."}`
- `EXAMPLE_INPUT = {"question": "List three programming languages created at Google.", "verified": []}`

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Question` → `Baseline answer` → `Plan checks` → `Verify Q1`, `Verify Q2` and `Verify Q3` (independent, never see the baseline) → `Revise` → `Final answer`.

**Test:**
```python
from agent_patterns.p13_chain_of_verification.graph import build_graph, mock_llm, EXAMPLE_INPUT

def test_cove_verifies_claims_and_revises():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["baseline"] == "Go, Dart and Python."
    assert len(out["checks"]) == 3
    assert dict(out["verified"])["Was Python created at Google?"].startswith("No")
    assert out["final"].startswith("Go and Dart")
    assert llm.calls == 6

def test_verification_is_factored_from_the_draft():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    verify_prompts = [p for p in llm.prompts if p.startswith("Answer briefly")]
    assert len(verify_prompts) == 3 and all(out["baseline"] not in p for p in verify_prompts)
```

**simple.py:** baseline → plan checks → `ThreadPoolExecutor().map(verify_one, checks)` → revise.

**Parity** (`test_simple_matches_graph`): `baseline`, `checks`, `sorted(s["verified"]) == sorted(g["verified"])`, `final` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The `Send` payload makes the factored isolation explicit: each verifier only receives its question.

**README facts:** Factored verification keeps the draft's mistakes from biasing the checks. **vs Self-Refine:** CoVe targets factual hallucinations claim by claim, where Self-Refine judges overall quality.

**README row:** `| 13 | [Chain-of-Verification](agent_patterns/p13_chain_of_verification/) | Self-correction | Draft, plan check questions, answer them independently, revise | 6 | Cutting factual hallucinations in lists/long answers | Creative tasks; latency-sensitive chat |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 15: 14 · Tree of Thoughts

**Files:** `agent_patterns/p14_tree_of_thoughts/` → `__init__.py` (empty), `graph.py`, `simple.py`, `test_graph.py`, `README.md`; plus the root README row

**Spec (Game of 24, BFS with beam):**
- A thought is `{"nums": list[float], "steps": list[str]}`. State: `frontier: list[dict]`, `candidates: list[dict]`, `depth: int`, `solution: list[str]`. `BEAM = 2`, `MAX_DEPTH = 3`. `SCORES = {"impossible": 0.0, "sure": 1.0, "likely": 0.5}`, checked in that order.
- `fmt(nums)` = `" ".join(f"{n:g}" for n in sorted(nums))`.
- `PROPOSE = "Game of 24. Propose up to 3 next steps, one per line, as 'a op b = c' using two of the numbers.\nNumbers: {nums}\n"`
- `EVALUATE = "Game of 24. Can these numbers reach 24? Answer sure, likely or impossible.\nRate: {nums}\n"`
- `apply_step(nums: list[float], line: str) -> dict | None`:
  - Match `^\s*(-?\d+(?:\.\d+)?)\s*([-+*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)`.
  - Remove `a`, then `b`, from a copy of `nums` (multiset). Return `None` if either is missing, on division by zero, or if `abs(result - claimed) > 1e-6`.
  - Otherwise return `{"nums": rest + [result], "step": f"{a:g} {op} {b:g} = {result:g}"}`.
  - The model's arithmetic is always re-checked in code, never trusted.
- Module-level helpers that the nodes wrap and `simple.py` reuses: `expand(frontier: list[dict], llm) -> tuple[list[dict], list[str] | None]` returns `(candidates, solution)`, and `score_and_prune(candidates: list[dict], llm) -> list[dict]` returns the new frontier.
- Nodes:
  - `generate`: sets `depth + 1`, then calls `expand`. For each frontier thought, one PROPOSE call. For each reply line, `apply_step`. A child whose `nums` is a single value within 1e-6 of 24 returns `{"solution": child steps}` immediately. Other children are deduped by `tuple(sorted(nums))`, first one seen kept, and go into `candidates`.
  - `evaluate`: calls `score_and_prune`. One EVALUATE call per candidate. Drop scores of 0, stable-sort by score descending, and keep the top `BEAM` as `frontier`.
- Edges: `START→generate`; `generate → END if solution else evaluate`; `evaluate → END if not frontier or depth >= MAX_DEPTH else generate`.
- Mock: `MockLLM(responses=["impossible"], by_keyword={"Numbers: 4 9 10 13\n": "10 - 4 = 6\n13 - 9 = 4\n4 + 9 = 13", "Numbers: 6 9 13\n": "13 - 9 = 4\n6 + 9 = 15", "Numbers: 4 4 10\n": "10 - 4 = 6\n4 * 4 = 16", "Numbers: 4 6\n": "6 * 4 = 24\n6 + 4 = 10", "Rate: 6 9 13\n": "likely", "Rate: 4 4 10\n": "likely", "Rate: 4 6\n": "sure"})`. Unmatched prompts fall back to `"impossible"`.
- `EXAMPLE_INPUT = {"frontier": [{"nums": [4, 9, 10, 13], "steps": []}], "depth": 0}`
- Expected trace: depth 1 makes 1 propose and 3 rates, keeping `6 9 13` and `4 4 10`. Depth 2 makes 2 proposes, finds `4 6` twice (deduped), and makes 3 rates, keeping `4 6`. Depth 3 makes 1 propose, which solves it. Total: 10 calls.

**Concept diagram** (for the folder README; labels ≤ 20 chars): `Numbers` → `Propose thoughts` → `Score each<br/>sure/likely/no` → `Keep best 2` → `{Reached 24?}` -no→ back to `Propose thoughts`; -yes→ `Solution`.

**Test:**
```python
from agent_patterns.core import MockLLM
from agent_patterns.p14_tree_of_thoughts.graph import build_graph, mock_llm, apply_step, EXAMPLE_INPUT

def test_tot_beam_search_finds_24():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["solution"] == ["10 - 4 = 6", "13 - 9 = 4", "6 * 4 = 24"]
    assert llm.calls == 10

def test_tot_gives_up_when_every_branch_is_impossible():
    llm = MockLLM(responses=["impossible"], by_keyword={"Numbers: 4 9 10 13\n": "10 - 4 = 6\n13 - 9 = 4"})
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out.get("solution") is None and out["frontier"] == [] and llm.calls == 3

def test_apply_step_rechecks_model_arithmetic():
    assert apply_step([4, 9, 10, 13], "10 - 4 = 6") == {"nums": [9, 13, 6.0], "step": "10 - 4 = 6"}
    assert apply_step([4, 9, 10, 13], "10 - 4 = 7") is None      # wrong math
    assert apply_step([4, 9, 10, 13], "5 + 5 = 10") is None      # numbers not available
    assert apply_step([4, 9, 10, 13], "no step here") is None
```

**simple.py:** a `for depth in range(MAX_DEPTH)` loop calling the shared `expand(frontier, llm)` and `score_and_prune(candidates, llm)` helpers, which the graph nodes also wrap.

**Parity** (`test_simple_matches_graph`): `solution` equal, and the same `llm.calls`.

**Plain vs graph** (README section): The plain BFS loop is shorter. The graph adds checkpointing of a long, expensive search and a trace of each level.

**README facts:** In the paper's abstract, Game of 24 went from 4% with GPT-4 CoT to 74% with ToT. The paper's four parts are thought decomposition, generator, state evaluator and search algorithm (BFS/DFS). Cost is 10–100× the LLM calls. **vs Chain-of-Thought and Reflexion:** ToT keeps many partial paths alive and prunes them while it reasons. That is search, not retry. LATS, the next lesson, adds MCTS, tools and reflection.

**README row:** `| 14 | [Tree of Thoughts](agent_patterns/p14_tree_of_thoughts/) | Search | Branch into candidate thoughts, score, prune, search (BFS/beam) | 10 | Puzzles/planning where partial steps can be scored | Simple Q&A; no way to score intermediate steps |`

- [ ] S1 - [ ] S2 - [ ] S3 - [ ] S4 - [ ] S5 - [ ] S6 - [ ] S7

---

### Task 16: README decision guide + OpenRouter smoke test

**Files:**
- Modify: `README.md` (add `## Pattern families`, `## Which pattern should I use?` and `## Plain Python or LangGraph?`)
- Modify: `tests/test_conventions.py` (add the flowchart check)
- Create: `tests/test_openrouter_smoke.py`

**Interfaces:** Consumes `PATTERNS`, each module's `build_graph`/`EXAMPLE_INPUT`, and each folder README's H1 `# NN · Name`.

- [ ] **Step 1: Add the failing check to `tests/test_conventions.py`**

```python
@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_is_in_decision_guide(name):
    title = (ROOT / "agent_patterns" / name / "README.md").read_text().splitlines()[0].split("· ", 1)[1]
    guide = (ROOT / "README.md").read_text().split("## Which pattern should I use?", 1)[1]
    assert title in guide, f"add '{title}' to the decision flowchart"
```

- [ ] **Step 2: Run it.** `uv run pytest tests/test_conventions.py -q`. Expected: 14 FAIL, because the section does not exist yet.

- [ ] **Step 3: Write the README sections.**
  - `## Pattern families` has 4 one-line entries: Tool agents (01, 02, 03, 06, 12), Workflows (07–11, which follow Anthropic's "Building effective agents"), Self-correction (04, 05, 13) and Search (14). Add one sentence: start with the simplest workflow that works, and move to agents only when the steps can't be known upfront.
  - `## Plain Python or LangGraph?`: one sentence saying every folder ships both (`simple.py` and `graph.py`, same prompts and mock, same result). Then 4 bullets on when the graph pays off:
    - parallel branches (`Send` and reducers: LLMCompiler, map-reduce, orchestrator, CoVe)
    - long or expensive runs that need checkpoint and resume
    - human-in-the-loop approval (`interrupt`)
    - observability (per-node trace, a diagram generated from the code, Studio/LangSmith)
    
    Close with one line: short loops (Self-Refine, Self-Ask, Evaluator-optimizer) read better as plain Python.
  - `## Which pattern should I use?` is a ```` ```mermaid flowchart TD ```` decision tree. Each of the 14 patterns is a leaf, labelled with its exact doc title. The branch questions, in order: are the steps known upfront (fixed sequence → Prompt chaining; one of N paths → Routing; same op on many inputs → Parallelization (map-reduce); subtasks only known at runtime → Orchestrator-workers)? If not: are tools needed (multi-hop fact lookup → Self-Ask; can plan all calls upfront: independent → LLMCompiler, sequential → ReWOO; long task that may need replanning → Plan-and-Execute; must react to each result → ReAct)? Is output quality the problem (external check exists → Reflexion; clear rubric → Evaluator-optimizer; style polish → Self-Refine; factual hallucinations → Chain-of-Verification)? Is it a hard search problem where partial steps can be scored → Tree of Thoughts?

- [ ] **Step 4: Run it.** `uv run pytest tests/test_conventions.py -q`. Expected: all pass.

- [ ] **Step 5: Write `tests/test_openrouter_smoke.py`**

```python
import importlib, os
import pytest
from agent_patterns.core import PATTERNS

pytestmark = pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")

@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_completes_with_real_model(name, monkeypatch):
    monkeypatch.setenv("USE_OPENROUTER", "1")
    mod = importlib.import_module(f"agent_patterns.{name}.graph")
    out = mod.build_graph().invoke(mod.EXAMPLE_INPUT, {"recursion_limit": 40})
    assert out    # finishes on real-model text (format fallbacks hold)
```

- [ ] **Step 6: Final verification.**
  - `uv run pytest -q`. Expected: all pass, with the 14 smoke tests skipped.
  - `uv run python -m agent_patterns.core`. Expected: prints `[]`, meaning no stale diagrams.
  - Link check every URL in the folder READMEs and the root README:
    ```bash
    grep -rhoE 'https?://[^) ]+' agent_patterns/p*/README.md README.md | sort -u | while read u; do echo "$(curl -sIL -o /dev/null -w '%{http_code}' "$u") $u"; done
    ```
    Expected: every line starts with `200`.
  - With a key, the user runs `uv run --env-file .env pytest tests/test_openrouter_smoke.py -q` themselves.

- [ ] **Step 7: Commit.** `git add -A && git commit -m "docs: add pattern families, decision guide and OpenRouter smoke test"`

- [ ] **Step 8: Ask the user** whether to create a GitHub repo and push (`gh repo create`). Do not push without a yes.
