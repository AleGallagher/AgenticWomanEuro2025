from fastapi import APIRouter, HTTPException

from dto.feedback_dto import FeedbackDto
from services.telegram_service import TelegramService

router = APIRouter()

@router.post("/feedback")
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
    telegram_service = TelegramService()
    telegram_service.send_feedback(feedback.feedback)