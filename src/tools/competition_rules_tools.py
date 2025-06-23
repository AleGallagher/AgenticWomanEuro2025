from langchain.tools import tool

COMPETITION_RULES_TEXT = """
UEFA Women's Euro 2025 Qualification Rules:

Group Stage:
- Teams in each group play each other once.
- 3 points for a win, 1 point for a draw, 0 points for a loss.
- The top two (2) teams from each group automatically qualify for the quarter-finals.
- If two or more teams are equal on points on completion of the group matches, the following tie-breaking criteria are applied in the order given:
    1. Higher number of points obtained in the matches played among the teams in question (head-to-head points).
    2. Superior goal difference resulting from the matches played among the teams in question (head-to-head goal difference).
    3. Higher number of goals scored in the matches played among the teams in question (head-to-head goals scored).
    4. If, after having applied criteria 1 to 3, teams still have an equal ranking, criteria 1 to 3 are reapplied exclusively to the matches between the remaining teams to determine their final rankings. If this procedure does not lead to a decision, criteria 5 to 9 apply to the group matches as a whole.
    5. Superior goal difference in all group matches.
    6. Higher number of goals scored in all group matches.
    7. Lower disciplinary points total based only on yellow and red cards received in all group matches (red card = 3 points, yellow card = 1 point, expulsion for two yellow cards in one match = 3 points).
    8. Higher position in the UEFA women's national team coefficient rankings used for the final tournament draw.
"""

@tool("CompetitionRulesTool", return_direct=True, description="Provides the text of the official qualification rules and tie-breaking criteria. Use this after getting current data from SQLQueryTool if you need to understand how qualification works for a specific scenario.")
def get_group_stage_rules():
    """
    Provides the text of the official qualification rules and tie-breaking criteria. Use this after getting current data from SQLQueryTool if you need to understand how qualification works for a specific scenario
    """
    return COMPETITION_RULES_TEXT
