from pydantic import BaseModel, Field
from typing import List

class QuestionMetadataOutput(BaseModel):
    """
    Structured output for extracting question metadata.
    """
    countries: List[str] = Field(description="A list of country names mentioned in the question. Empty list if none.")