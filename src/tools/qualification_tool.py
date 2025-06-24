from langchain.tools import tool
from agents.qualification_agent import handle_qualification_question

@tool("qualification_tool", return_direct=True, description="Use to know what need a team to qualify, pass to the next stage. Example: 'What need Spain to qualify?', or 'What need Spain to qualify to the next stage?'")
def get_qualification_options(model, agent_input: str = "", question_language: str = "English"):
    """
    Use to know what need a team to qualify, pass to the next stage. Example: 'What need Spain to qualify?', or 'What need Spain to qualify to the next stage?'"
    """
    result = handle_qualification_question(model, agent_input, question_language)
    return result.content.strip()
