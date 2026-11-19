"""17 · Hierarchical teams: a top supervisor (CEO) picks a team; each team's own lead picks its members."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

FINISH = "FINISH"          # the CEO is done with the whole task
DONE = "DONE"              # a team lead is done with the team's part
MAX_TEAM_TURNS = 4         # team runs the CEO may order
MAX_WORKER_TURNS = 4       # member runs a lead may order per team run
NO_ANSWER = "(no answer)"

# The company: each team has a short description and its own members (each member = its own prompt).
TEAMS = {
    "research": {
        "about": "market and pricing facts",
        "workers": {
            "market_researcher": "You are the market researcher. Find who will buy the product and what competitors "
                                 "charge. Reply with the facts only.",
            "data_analyst": "You are the data analyst. Turn the facts into a launch price and show the math. "
                            "Reply with the result only.",
        },
    },
    "marketing": {
        "about": "the launch message",
        "workers": {
            "copywriter": "You are the copywriter. Write a short launch tagline from the company work so far.",
            "editor": "You are the editor. Polish the copy into one final launch line with the price. "
                      "Reply with that line only.",
        },
    },
}

CEO = """You are the CEO. Your teams: {teams}.
Read the company work so far and reply with ONE word: the team that should work next, or FINISH when the task is done.

Task: {task}
Company work so far:
{company}"""

LEAD = """You lead the {team} team. Team members: {members}.
Reply with ONE word: the member who should work next, or DONE when your team's part is finished.

Task: {task}
Company work so far:
{company}
Team work so far:
{team_work}"""

WORKER = """{role}

Task: {task}
Company work so far:
{company}
Team work so far:
{team_work}"""


class State(TypedDict):                     # the CEO's level
    task: str
    history: Annotated[list[tuple[str, str]], operator.add]   # (team, report)
    next_team: str
    turns: int
    latest: str                             # newest member output, the answer when the CEO finishes
    answer: str


class TeamState(TypedDict):                 # one team's level - a different, private schema
    task: str
    company: str                            # read-only view of the other teams' reports
    team_history: Annotated[list[tuple[str, str]], operator.add]   # (member, output)
    next_worker: str
    worker_turns: int
    report: str
    latest: str


def work_text(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in history) or "(nothing yet)"


def parse_choice(reply: str, options, stop: str) -> str:
    """A known option, or `stop`. Unrecognized replies end that level instead of looping on a guess."""
    word = re.sub(r"[^a-z_]", "", reply.lower())
    return word if word in options else stop


def ceo_decide(task: str, history: list[tuple[str, str]], turns: int, llm: BaseChatModel) -> str:
    if turns >= MAX_TEAM_TURNS:
        return FINISH
    teams = ", ".join(f"{name} ({t['about']})" for name, t in TEAMS.items())
    reply = llm.invoke(CEO.format(teams=teams, task=task, company=work_text(history))).content
    return parse_choice(str(reply), TEAMS, FINISH)


def lead_decide(team: str, task: str, company: str, team_history: list[tuple[str, str]], turns: int,
                llm: BaseChatModel) -> str:
    if turns >= MAX_WORKER_TURNS:
        return DONE
    members = ", ".join(TEAMS[team]["workers"])
    reply = llm.invoke(LEAD.format(team=team, members=members, task=task, company=company,
                                   team_work=work_text(team_history))).content
    return parse_choice(str(reply), TEAMS[team]["workers"], DONE)


def member_work(team: str, worker: str, task: str, company: str, team_history: list[tuple[str, str]],
                llm: BaseChatModel) -> str:
    prompt = WORKER.format(role=TEAMS[team]["workers"][worker], task=task, company=company,
                           team_work=work_text(team_history))
    return str(llm.invoke(prompt).content).strip() or "(empty reply)"


def mock_llm() -> MockLLM:
    # every level takes turns, so one plain sequence is deterministic
    return MockLLM(responses=[
        "research",                                   # CEO
        "market_researcher",                          # research lead
        "Buyers: gym-goers aged 25-40. Competitor bottles sell for $35-$45.",
        "data_analyst",                               # research lead
        "Launch at $39: inside the $35-$45 range, about 30% margin on a $27 unit cost.",
        "DONE",                                       # research lead
        "marketing",                                  # CEO
        "copywriter",                                 # marketing lead
        "HydroSmart reminds gym-goers to drink, for $39.",
        "editor",                                     # marketing lead
        "HydroSmart: the smart bottle that keeps you hydrated. Launch price $39.",
        "DONE",                                       # marketing lead
        "FINISH",                                     # CEO
    ])


def build_team_graph(team: str, llm: BaseChatModel):
    """One team = its own small supervisor graph: lead -> member -> lead ... -> DONE."""
    def lead(state: TeamState):
        choice = lead_decide(team, state["task"], state["company"], state["team_history"], state["worker_turns"], llm)
        if choice == DONE:
            history = state["team_history"]
            return {"next_worker": DONE, "report": work_text(history), "latest": history[-1][1] if history else ""}
        return {"next_worker": choice}

    def make_member(worker: str):
        def member(state: TeamState):
            output = member_work(team, worker, state["task"], state["company"], state["team_history"], llm)
            return {"team_history": [(worker, output)], "worker_turns": state["worker_turns"] + 1}
        return member

    g = StateGraph(TeamState)
    g.add_node("lead", lead)
    for worker in TEAMS[team]["workers"]:
        g.add_node(worker, make_member(worker))
        g.add_edge(worker, "lead")
    g.add_edge(START, "lead")
    g.add_conditional_edges("lead", lambda s: END if s["next_worker"] == DONE else s["next_worker"],
                            {**{w: w for w in TEAMS[team]["workers"]}, END: END})
    return g.compile()


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm
    team_graphs = {team: build_team_graph(team, llm) for team in TEAMS}

    def ceo(state: State):
        choice = ceo_decide(state["task"], state["history"], state["turns"], llm)
        if choice == FINISH:
            return {"next_team": FINISH, "answer": state.get("latest") or NO_ANSWER}
        return {"next_team": choice}

    def make_team_node(team: str):
        def run_team(state: State):
            # the team graph has its own state schema, so the parent calls it and maps the result back
            out = team_graphs[team].invoke({"task": state["task"], "company": work_text(state["history"]),
                                            "team_history": [], "worker_turns": 0})
            update = {"history": [(team, out["report"])], "turns": state["turns"] + 1}
            return {**update, "latest": out["latest"]} if out.get("latest") else update
        return run_team

    g = StateGraph(State)
    g.add_node("ceo", ceo)
    for team in TEAMS:
        g.add_node(team, make_team_node(team))
        g.add_edge(team, "ceo")                 # every team reports back to the CEO
    g.add_edge(START, "ceo")
    g.add_conditional_edges("ceo", lambda s: END if s["next_team"] == FINISH else s["next_team"],
                            {**{t: t for t in TEAMS}, END: END})
    return g.compile()


EXAMPLE_INPUT = {"task": "Plan the launch of HydroSmart, our new smart water bottle: who buys it, the price, "
                         "and the launch line.",
                 "history": [], "turns": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
