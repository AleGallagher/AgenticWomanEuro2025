from langchain.tools import tool
from agents.sql_agent import SQLAgent

@tool("SQLQueryTool", return_direct=True, description="Use for specific data such as coach names, team information, player lists, scores, and matches")
def get_sql_tool(model, agent_input: str = "", question_language: str = "English"):
    """
    Use for specific data such as coach names, team information, player lists, scores, and matches
    """
    initial_state = {
    "input": agent_input,
    "question_language": question_language,
    }
    graph = SQLAgent(model)
    final_state = graph.graph.invoke(initial_state)
    return final_state["messages"][-1].content
