
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rag.agentic_rag import AgenticRAG

@tool("agentic_rag", return_direct=True, description="Use this for general, historical knowledge questions or open-ended information about the Women's Eurocup 2025. Example: 'What can you say about Spain?' or 'Is there VAR?' or 'When will start the cup? or Top Goal Scorers in the hostory'")
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
     