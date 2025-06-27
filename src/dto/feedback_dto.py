from pydantic import BaseModel

class FeedbackDto(BaseModel):
    """
    A Data Transfer Object (DTO) for sending feedback.

    Attributes:
        feedback (str): feedback text provided by the user.
    """
    feedback: str
    
    def __repr__(self):
        return f"FeedbackDto(feedback={self.feedback})"