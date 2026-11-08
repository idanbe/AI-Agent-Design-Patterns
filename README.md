# Agent Patterns in LangGraph

AI agent design patterns, one folder each: a runnable LangGraph graph, the same pattern in plain Python, and a README with diagrams and a simple explanation.
The curriculum follows the "⚡ Daily agent pattern lesson" routine — one new pattern a day.

## Quickstart

```bash
uv sync
uv run pytest
uv run python -m agent_patterns.p01_react.graph    # LangGraph version, prints each node as it runs
uv run python -m agent_patterns.p01_react.simple   # same pattern in plain Python
```

Everything runs offline on a scripted mock LLM — no API key needed.

## Real model (OpenRouter)

```bash
cp .env.example .env   # put your OPENROUTER_API_KEY in .env
USE_OPENROUTER=1 uv run --env-file .env python -m agent_patterns.p01_react.graph
```

Pick another model with `OPENROUTER_MODEL` (default `anthropic/claude-haiku-5.5`).

## Patterns

| # | Pattern | Family | Idea | LLM calls* | Use when | Avoid when |
|---|---|---|---|---|---|---|
| 01 | [ReAct](agent_patterns/p01_react/) | Tool agent | Think → act → observe loop, one LLM call per step | 3 | Steps unknown upfront; must react to tool results | Fixed workflows; tight latency/cost budgets |
| 02 | [ReWOO](agent_patterns/p02_rewoo/) | Tool agent | Plan every tool call once with #E variables, run them, then solve | 2 | Predictable multi-tool lookups; fewer tokens than ReAct | Later steps depend on surprises in earlier results |
| 03 | [Plan-and-Execute](agent_patterns/p03_plan_execute/) | Tool agent | Planner writes steps, executor does one, replanner adjusts | 5 | Long multi-step tasks; strong planner + cheap executor | Short tasks (planning overhead); hard real-time |
| 04 | [Reflexion](agent_patterns/p04_reflexion/) | Self-correction | Retry with verbal lessons learned from an external evaluator | 3 | You have a ground-truth check (tests, rules, env reward) | No reliable evaluator; one-shot tasks |

\* LLM calls in the repo's mock example — a rough cost/latency signal.

## Adding a pattern

In Claude Code run `/add-pattern <name>` — the recipe lives in [.claude/skills/add-pattern/SKILL.md](.claude/skills/add-pattern/SKILL.md).
`tests/test_conventions.py` fails until the new folder, its diagrams and its table row are all in place.
