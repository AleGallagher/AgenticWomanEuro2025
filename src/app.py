import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from agents.main_agent import MainAgent
from config.dependencies import get_model, get_store
from config.errors.exceptions import InvalidRequestException
from config.errors.handlers import register_exception_handlers
from config.logging_config import setup_logging
from dto.feedback_dto import FeedbackDto
from dto.message_dto import MessageDto
from services.telegram_service import TelegramService

load_dotenv()

logger = logging.getLogger(__name__)
app = FastAPI()
telegram_service = TelegramService()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
setup_logging()
register_exception_handlers(app)
agent = MainAgent(model=get_model(), vector_store=get_store())

@app.get("/")
async def root():
    return {"greeting": "Hello UEFA Women's EURO 2025"}

@app.post("/message")
async def sendMessage(
    message: MessageDto
):
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
        result = agent(state=initial_state, config=config)
        return {"output": result["messages"][-1].content}
    except Exception as e:
        print(e)
        print(f"Error while invoking agent executor: {e}")
        return {"output": "Sorry, I cannot answer this question now. Please try a different request or rephrase your question."}

@app.post("/feedback")
async def sendFeedback(feedback: FeedbackDto):
    """
    Handles feedback submission and sends it via email.

    Args:
        feedback (FeedbackDto): The feedback data containing the feedback text.

    Returns:
        dict: A success message or an error message.
    """
    if not feedback.feedback.strip():
        raise HTTPException(status_code=400, detail="The 'feedback' field cannot be empty.")
    telegram_service.send_feedback(feedback.feedback)
