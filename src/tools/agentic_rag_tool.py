
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rag.agentic_rag import AgenticRAG

@tool("agentic_rag", return_direct=True, description=(
                    "Use this tool for general background, historical knowledge, or open-ended questions "
                    "about the Women's Eurocup 2025. This includes rules (e.g., VAR), past tournaments, hosts, and top scorers in history. "
                    "Examples:\n"
                    "- 'What can you say about Spain?'\n"
                    "- 'Is there VAR?'\n"
                    "- 'When does the cup start?'\n"
                    "- 'Top goal scorers in tournament history?'"
                )
)
def agentic_rag_stream(vector_store, question: str = "", language: str = "English"):
    """
    Retrieve relevant documents based on the query.
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question_language": language,
        "agent_action": "",
        "rewrite_count": 0,
    }
    argentic_rag = AgenticRAG(vector_store)
    return argentic_rag.graph.invoke(initial_state)["messages"][-1].content
     