from typing import List

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word ansert to the question.")
    reflection: Reflection = Field(description="Your reflection on the answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvem ents to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    """Revise the answer based on the critique and new information."""

    references: List[str] = Field(
        description="References for the revised answer in the form of URLs."
    )