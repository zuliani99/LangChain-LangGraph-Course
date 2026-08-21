"""
Agentic RAG / Adaptive RAG - hallucination grader.

Gate #1 on the edge leaving GENERATE: "is this answer actually supported by the
documents it was given?" It compares the generation against the facts only - it
has no opinion on whether the answer is useful, which is the answer grader's job.

A "no" sends the graph back to GENERATE for another attempt.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    # A real bool, so graph.py can branch on it directly. The description still
    # says "'yes' or 'no'" because that phrasing is what the model reads - the
    # JSON type coercion to true/false happens underneath.
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
# Note what is NOT asked: nothing about correctness in the world, only about
# support by {documents}. Groundedness is checkable; truth is not.
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
# answer yes or no question: is the answer grounded in the facts?