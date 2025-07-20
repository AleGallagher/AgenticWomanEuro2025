from typing import Annotated
from langchain.tools import tool
from agents.sql_agent import SQLAgent
from langchain_core.tools import InjectedToolArg
from langchain_openai import ChatOpenAI

@tool("SQLQueryTool", return_direct=True, description=(
                    "Use this tool for specific, structured data. "
                    "Examples include: coach names, team information, player lists, match schedules, scores, and live tournament stats."
                    "Also use this for **statistics or aggregates** like totals, averages, and per-match metrics. "
                    "Example questions:\n"
                    "- 'Who is the coach of England?'\n"
                    "- 'Who plays today?'\n"
                    "- 'Which players scored the most goals?'\n"
                    "- 'How many substitutions happen on average per match?'\n"
                    "- 'Total goals by Spain?'"
                )
)
def get_sql_tool(model: Annotated[ChatOpenAI, InjectedToolArg], question: str = "", question_language: Annotated[str, InjectedToolArg] = "English"):
    """
    Use for specific data such as coach names, team information, player lists, scores, and matches
    """
    initial_state = {
    "input": question,
    "question_language": question_language,
    }
    graph = SQLAgent(model)
    final_state = graph.graph.invoke(initial_state)
    return final_state["messages"][-1].content
