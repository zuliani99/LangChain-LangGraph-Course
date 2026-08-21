"""
Agentic RAG / Adaptive RAG - answer grader.

Gate #2 on the edge leaving GENERATE, evaluated only after the hallucination
grader has passed: "grounded, fine - but does it actually answer the question?"

A perfectly grounded answer can still miss the point (right documents, wrong
aspect). A "no" here routes to WEB_SEARCH to widen the evidence, rather than
back to GENERATE - regenerating from the same insufficient context would not help.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Note the inputs: {question} and {generation} only - the documents are
# deliberately absent, so this grader cannot be swayed by how well-sourced the
# answer looks and judges relevance to the question alone.
answer_grader = answer_prompt | structured_llm_grader
# answer yes or no question: does the answer address the question?