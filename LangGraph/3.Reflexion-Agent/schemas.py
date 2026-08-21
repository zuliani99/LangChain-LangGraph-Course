"""
LangGraph lesson 3 / schemas.py - the contract the actor must fill in.

These Pydantic models are passed to bind_tools(), so they are not just
validation: they become the JSON schema of a forced tool call. Making the
critique a REQUIRED FIELD is what stops the model from skipping self-criticism -
it literally cannot emit a valid answer without also emitting its own critique
and the search queries needed to fix it.
"""

from typing import List

from pydantic import BaseModel, Field

# Two-sided critique: what to ADD and what to CUT. Asking for both keeps the
# answer from growing without bound over successive revisions.
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


# Tool used for the FIRST draft. The docstring becomes the tool description and
# the class name becomes the tool name ("AnswerQuestion") - which is why
# tool_executor.py registers its executor under exactly that name.
class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word ansert to the question.")
    reflection: Reflection = Field(description="Your reflection on the answer.")
    # The self-generated research plan: these strings are what tool_executor.py
    # actually sends to Tavily on the next node.
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvem ents to address the critique of your current answer."
    )

# Tool used for every REVISION. Inherits answer/reflection/search_queries and
# adds the mandatory citations, so a revised answer must be verifiable.
class ReviseAnswer(AnswerQuestion):
    """Revise the answer based on the critique and new information."""

    references: List[str] = Field(
        description="References for the revised answer in the form of URLs."
    )