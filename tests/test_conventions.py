"""Every pattern folder follows the same shape, so a new pattern each day stays consistent."""
import importlib
import re
from pathlib import Path

import pytest

from agent_patterns.core import PATTERNS, mermaid_for

ROOT = Path(__file__).resolve().parents[1]
MERMAID = re.compile(r"<!-- mermaid:start -->\n```mermaid\n(.*?)\n```\n<!-- mermaid:end -->", re.S)
LABEL = re.compile(r'[\[({]+"?([^"\])}]+?)"?[\])}]+')     # node text inside [..] (..) {..}
HEADINGS = ["## In plain words", "## How it works", "## The LangGraph graph", "## Plain Python vs LangGraph",
            "## Run it", "## When to use", "## When not to use", "## vs. closest pattern", "## Sources"]


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
    # link the README file itself: local Markdown previews can't open a folder link (GitHub can)
    assert f"(agent_patterns/{name}/README.md)" in (ROOT / "README.md").read_text(), "add a root README table row"
    mod.build_graph(mod.mock_llm()).invoke(mod.EXAMPLE_INPUT)   # example runs offline
    assert "langgraph" not in (folder / "simple.py").read_text(), "simple.py must be plain Python"
    importlib.import_module(f"agent_patterns.{name}.simple").run(mod.EXAMPLE_INPUT, mod.mock_llm())


@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_is_in_decision_guide(name):
    title = (ROOT / "agent_patterns" / name / "README.md").read_text().splitlines()[0].split("· ", 1)[1]
    guide = (ROOT / "README.md").read_text().split("## Which pattern should I use?", 1)[1]
    assert title in guide, f"add '{title}' to the decision flowchart"
