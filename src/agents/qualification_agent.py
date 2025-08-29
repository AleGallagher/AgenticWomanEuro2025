from langchain.prompts import PromptTemplate

from services.prompt_utils import PromptUtils
from tools.sql_tool import get_sql_tool

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

def handle_qualification_question(llm, question, question_language):
    """
    Handle a user question about team qualification scenarios in the Women's Eurocup 2025.

    This function retrieves current group standings and upcoming matches using a SQL tool,
    fetches qualification rules using a rules tool, and synthesizes the information with
    a language model to provide a detailed answer about what a team needs to do to qualify.

    Args:
        question (str): The user's question about team qualification.
        llm: The language model instance used for generating the final answer.

    Returns:
        str: A detailed explanation of the qualification scenarios for the team in question.
    """
    # Step 1: Use SQLQueryTool to get current standings and upcoming matches
    result = get_sql_tool.invoke({"model": llm, "agent_input": f"Get current group standings and upcoming matches for the group of the team mentioned in: '{question}'", "question_language": "English"})
    sql_result = result

    rules_result = COMPETITION_RULES_TEXT
    # Step 2: Load prompt template from YAML
    prompt_config = PromptUtils.load_prompt_template("qualification_analysis", "v0") 
    # Step 3: Synthesize with LLM
    prompt_template = PromptTemplate(
        input_variables=prompt_config["input_variables"],
        template=prompt_config["template"]
    )

    llm_chain = prompt_template | llm
    answer = llm_chain.invoke({
        "question": question,
        "sql_data": sql_result,
        "rules": rules_result,
        "language": question_language,
    })

    return answer