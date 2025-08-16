
from typing import Annotated

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg

from rag.agentic_rag import AgenticRAG
from rag.vector_stores.base_store import BaseStore

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
def agentic_rag(vector_store: Annotated[BaseStore, InjectedToolArg], question: str = "", question_language: Annotated[str, InjectedToolArg] = "English"):
    """
    Retrieve relevant answer based on the query.
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question_language": question_language
    }
    argentic_rag = AgenticRAG(vector_store)
    return argentic_rag(initial_state)["messages"][-1].content
     