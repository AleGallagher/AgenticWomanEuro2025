from tools.sql_tool import get_sql_tool
from langchain.prompts import PromptTemplate

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
"""

def handle_qualification_question(llm, user_input, question_language):
    """
    Handle a user question about team qualification scenarios in the Women's Eurocup 2025.

    This function retrieves current group standings and upcoming matches using a SQL tool,
    fetches qualification rules using a rules tool, and synthesizes the information with
    a language model to provide a detailed answer about what a team needs to do to qualify.

    Args:
        user_input (str): The user's question about team qualification.
        llm: The language model instance used for generating the final answer.

    Returns:
        str: A detailed explanation of the qualification scenarios for the team in question.
    """
    # Step 1: Use SQLQueryTool to get current standings and upcoming matches
    result = get_sql_tool.invoke({"model": llm, "agent_input": f"Get current group standings and upcoming matches for the group of the team mentioned in: '{user_input}'", "question_language": "English"})
    sql_result = result

    rules_result = COMPETITION_RULES_TEXT

    # Step 3: Synthesize with LLM
    prompt_template = PromptTemplate(
        input_variables=["team", "current_standings", "remaining_matches", "rules"],
        template="""
        You are an expert on the Women's Eurocup 2025.

        The user asked: {question}

        You retrieved the following data from the database:
        {sql_data}

        You also retrieved the following qualification rules:
        {rules}

    

        Now, based on the standings paying attention on the matches won, lost and draw, the points of each team, upcoming matches, and rules, explain what the team needs to do to qualify.
        ONLY use the remaining group matches and the current status of the group. DO NOT include already played matches in the analysis.
        Qualification depends on possible outcomes of upcoming group matches only.
        Be specific. Mention possible scenarios (e.g., win + X team draws, or tie-breakers).
        1. If qualification is **mathematically impossible**, explain this clearly and briefly. Do NOT describe hypothetical or impossible scenarios.
        2. If qualification is **still possible**, provide a clear and specific explanation of what needs to happen:
        - Include required match outcomes and tie-breaker situations.
        - Use current standings, points, goal difference, and match data.
        - Only consider the remaining group matches. Ignore matches that have already been played.
        Be clear, and answer in {language} language.

        Final Answer:"""
    )

    llm_chain = prompt_template | llm
    answer = llm_chain.invoke({
        "question": user_input,
        "sql_data": sql_result,
        "rules": rules_result,
        "language": question_language
    })

    return answer