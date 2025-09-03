from functools import lru_cache

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage

from agents.main_agent import MainAgent
from config.dependencies import get_model, get_store
from config.errors.exceptions import InvalidRequestException
from dto.message_dto import MessageDto

router = APIRouter()

@lru_cache()
def provide_agent() -> MainAgent:
    return MainAgent(model=get_model(), vector_store=get_store())

@router.post("/message")
async def sendMessage(
    message: MessageDto,
    agent: MainAgent = Depends(provide_agent),
) -> dict:
    """
    Handles user messages and invokes the agent's graph.

    Args:
        message (MessageDto): The message data containing question, session_id, and country.
        vector_store: Dependency for vector store.
        model: Dependency for the model.

    Returns:
        dict: The result of the agent's graph invocation or an error message.
    """
    if not message.question.strip():
        raise InvalidRequestException("The 'question' field cannot be empty.")
    if not message.session_id.strip():
        raise InvalidRequestException("The 'session_id' field cannot be empty.")
    print(f"Question: {message} - rephrased_question: {message.question}")
    try:
        initial_state = {
            "messages": [HumanMessage(content=message.question)],
            "user_id": message.session_id,
            "country": message.country,
        }
        config = {"configurable": {"thread_id": message.session_id}}
        result = await agent(state=initial_state, config=config)
        return {"output": result["messages"][-1].content}
    except Exception as e:
        print(e)
        print(f"Error while invoking agent executor: {e}")
        return {"output": "Sorry, I cannot answer this question now. Please try a different request or rephrase your question."}
