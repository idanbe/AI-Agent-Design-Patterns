"""17 · Hierarchical teams in plain Python: same prompts, teams and mock as graph.py; only the control flow differs."""
from langchain_core.language_models import BaseChatModel

from agent_patterns.core import get_llm
from agent_patterns.p17_hierarchical_teams.graph import (FINISH, DONE, NO_ANSWER, ceo_decide, lead_decide, member_work,
                                                         work_text, mock_llm, EXAMPLE_INPUT)


def run_team(team: str, task: str, company: str, llm: BaseChatModel) -> tuple[str, str]:
    """Inner loop - one team: the lead picks members until DONE. Returns (report, latest output)."""
    team_history, turns = [], 0
    while (worker := lead_decide(team, task, company, team_history, turns, llm)) != DONE:
        team_history.append((worker, member_work(team, worker, task, company, team_history, llm)))
        turns += 1
    return work_text(team_history), team_history[-1][1] if team_history else ""


def run(inputs: dict, llm: BaseChatModel | None = None) -> dict:
    llm = get_llm(mock_llm) if llm is None else llm
    task, history, turns, latest = inputs["task"], [], 0, ""
    while (team := ceo_decide(task, history, turns, llm)) != FINISH:      # outer loop - the CEO picks a team
        report, team_latest = run_team(team, task, work_text(history), llm)
        history.append((team, report))
        latest = team_latest or latest
        turns += 1
    return {"answer": latest or NO_ANSWER, "history": history, "turns": turns}


if __name__ == "__main__":
    print(run(EXAMPLE_INPUT))
