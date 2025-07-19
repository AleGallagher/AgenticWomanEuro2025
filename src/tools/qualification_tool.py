from typing import Annotated
from langchain.tools import tool
from agents.qualification_agent import handle_qualification_question
from langchain_core.tools import InjectedToolArg
from langchain_openai import ChatOpenAI

@tool("qualification_tool", return_direct=True, description="Use to know what need a team to qualify, pass to the next stage. Example: 'What need Spain to qualify?', or 'What need Spain to qualify to the next stage?'")
def get_qualification_options(model: Annotated[ChatOpenAI, InjectedToolArg], question: str = "", question_language: Annotated[str, InjectedToolArg] = "English"):
    """
    Use to know what need a team to qualify, pass to the next stage. Example: 'What need Spain to qualify?', or 'What need Spain to qualify to the next stage?'"
    """
    result = handle_qualification_question(model, question, question_language)
    return result.content.strip()
