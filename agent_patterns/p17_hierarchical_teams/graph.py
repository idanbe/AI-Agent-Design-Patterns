"""17 · Hierarchical teams: a top supervisor (CEO) picks a team; each team's own lead picks its members."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agent_patterns.core import MockLLM, get_llm, run_demo

FINISH = "FINISH"          # the CEO is done with the whole task
DONE = "DONE"              # a team lead is done with the team's part
MAX_TEAM_TURNS = 8         # team runs the CEO may order (a team can be called again)
MAX_WORKER_TURNS = 4       # member runs a lead may order per team run
NO_ANSWER = "(no answer)"

# The company: each team has a short description and its own members (each member = its own prompt).
TEAMS = {
    "design": {
        "about": "how the feature works and looks",
        "workers": {
            "ux_designer": "You are the UX designer. Design the user flow for the feature. One or two sentences.",
            "ui_designer": "You are the UI designer. Describe how the feature looks on screen, based on the flow. "
                           "One or two sentences.",
        },
    },
    "development": {
        "about": "building it",
        "workers": {
            "backend_dev": "You are the backend developer. Plan the server side for the design: APIs, data, effort. "
                           "One or two sentences.",
            "frontend_dev": "You are the frontend developer. Plan how to build the screens from the design. "
                            "One or two sentences.",
        },
    },
    "marketing": {
        "about": "the launch campaign",
        "workers": {
            "copywriter": "You are the copywriter. Write the launch headline for the feature. One line.",
            "growth_marketer": "You are the growth marketer. Plan the launch campaign with a measurable goal. "
                               "One or two sentences.",
        },
    },
    "sales": {
        "about": "selling it to business customers",
        "workers": {
            "account_exec": "You are the account executive. Write the pitch for existing business customers. "
                            "One or two sentences.",
            "sales_engineer": "You are the sales engineer. Prepare the technical demo for customers. "
                              "One or two sentences.",
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
    answer: str                             # the launch plan: every team's report


class TeamIO(TypedDict):                    # what a team shares with the CEO's graph
    task: str
    history: list[tuple[str, str]]          # in: every team's report so far; out: just this team's new report
    turns: int                              # in: the CEO's team-turn count; out: one more


class TeamState(TeamIO):                    # plus the team's private working state, never seen by the CEO
    team_history: Annotated[list[tuple[str, str]], operator.add]   # (member, output)
    next_worker: str
    worker_turns: int


def work_text(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in history) or "(nothing yet)"


def company_text(history: list[tuple[str, str]]) -> str:
    """Every team's report, one section per team: what the CEO and all other teams see."""
    return "\n".join(f"[{team}]\n{report}" for team, report in history) or "(nothing yet)"


def parse_choice(reply: str, options, stop: str) -> str:
    """A known option, or `stop`. Unrecognized replies end that level instead of looping on a guess."""
    word = re.sub(r"[^a-z_]", "", reply.lower())
    return word if word in options else stop


def ceo_decide(task: str, history: list[tuple[str, str]], turns: int, llm: BaseChatModel) -> str:
    if turns >= MAX_TEAM_TURNS:
        return FINISH
    teams = ", ".join(f"{name} ({t['about']})" for name, t in TEAMS.items())
    reply = llm.invoke(CEO.format(teams=teams, task=task, company=company_text(history))).content
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
        "design",                                     # CEO
        "ux_designer",                                # design lead
        "Flow: the user types a task, the app suggests a reminder time from their calendar, one tap accepts it.",
        "ui_designer",                                # design lead
        "A reminder chip under each task with a bell icon; swipe to snooze.",
        "DONE",                                       # design lead
        "development",                                # CEO
        "backend_dev",                                # development lead
        "Reminder API and a job queue that reads the user's calendar; about two sprints.",
        "frontend_dev",                               # development lead
        "Build the reminder chip and swipe to snooze from the design; ship behind a feature flag.",
        "DONE",                                       # development lead
        "marketing",                                  # CEO
        "copywriter",                                 # marketing lead
        "Smart Reminders: TaskFlow now tells you when, not just what.",
        "growth_marketer",                            # marketing lead
        "Launch email to 20k users plus an in-app banner; goal: 15% of users try it in month one.",
        "DONE",                                       # marketing lead
        "sales",                                      # CEO
        "account_exec",                               # sales lead - one member is enough here
        "Pitch to our 40 team accounts: fewer missed deadlines; offer Smart Reminders in the Pro plan.",
        "DONE",                                       # sales lead
        "FINISH",                                     # CEO
    ])


def build_team_graph(team: str, llm: BaseChatModel):
    """One team = its own small supervisor graph: lead -> member -> lead ... -> DONE -> report."""
    def lead(state: TeamState):
        choice = lead_decide(team, state["task"], company_text(state["history"]), state.get("team_history", []),
                             state.get("worker_turns", 0), llm)
        return {"next_worker": choice}

    def make_member(worker: str):
        def member(state: TeamState):
            output = member_work(team, worker, state["task"], company_text(state["history"]),
                                 state.get("team_history", []), llm)
            return {"team_history": [(worker, output)], "worker_turns": state.get("worker_turns", 0) + 1}
        return member

    def report(state: TeamState):
        # overwrite `history` with only this team's report: the CEO's graph appends it to the company history
        return {"history": [(team, work_text(state.get("team_history", [])))], "turns": state["turns"] + 1}

    g = StateGraph(TeamState, input_schema=TeamIO, output_schema=TeamIO)
    g.add_node("lead", lead)
    g.add_node("report", report)
    for worker in TEAMS[team]["workers"]:
        g.add_node(worker, make_member(worker))
        g.add_edge(worker, "lead")
    g.add_edge(START, "lead")
    g.add_conditional_edges("lead", lambda s: "report" if s["next_worker"] == DONE else s["next_worker"],
                            {**{w: w for w in TEAMS[team]["workers"]}, "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def build_graph(llm: BaseChatModel | None = None):
    llm = get_llm(mock_llm) if llm is None else llm
    team_graphs = {team: build_team_graph(team, llm) for team in TEAMS}

    def ceo(state: State):
        choice = ceo_decide(state["task"], state["history"], state["turns"], llm)
        if choice == FINISH:
            return {"next_team": FINISH, "answer": company_text(state["history"]) if state["history"] else NO_ANSWER}
        return {"next_team": choice}

    g = StateGraph(State)
    g.add_node("ceo", ceo)
    for team in TEAMS:
        g.add_node(team, team_graphs[team])     # a compiled graph as a node: a real subgraph LangGraph can draw
        g.add_edge(team, "ceo")                 # every team reports back to the CEO
    g.add_edge(START, "ceo")
    g.add_conditional_edges("ceo", lambda s: END if s["next_team"] == FINISH else s["next_team"],
                            {**{t: t for t in TEAMS}, END: END})
    return g.compile()


EXAMPLE_INPUT = {"task": "Launch Smart Reminders, a new feature in TaskFlow, our task-management app: "
                         "design it, build it, market it and sell it.",
                 "history": [], "turns": 0}

if __name__ == "__main__":
    run_demo(build_graph(), EXAMPLE_INPUT)
