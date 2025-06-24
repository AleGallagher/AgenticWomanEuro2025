from pydantic import BaseModel
from typing import Optional

class MessageDto(BaseModel):
    """
    A Data Transfer Object (DTO) for handling messages.

    Attributes:
        question (str): The question being asked.
        session_id (str): The session ID associated with the chat.
        country (Optional[str]): The country associated with the message.
    """
    question: str
    session_id: str
    country: Optional[str] = None
    
    def __repr__(self):
        return f"MessageDto(question={self.question!r}, session_id={self.session_id!r}, country={self.country!r})"