"""
Agentic RAG / Adaptive RAG - retrieval grader chain.

Answers one question per document: "is this chunk relevant to the user's
question, yes or no?" Used by the GRADE_DOCUMENTS node to filter the retriever
output and to decide whether a web-search fallback is needed.

This is the first of the LLM-as-a-judge chains; the pattern (tiny Pydantic
schema + with_structured_output + a one-job system prompt) repeats identically
in the hallucination, answer and router chains.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# temperature=0: a grader must be reproducible. A cheap model is fine here -
# the judgement is binary and the call happens once per retrieved chunk.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    # NOTE: typed `str` and compared with .lower() == "yes" downstream. The other
    # graders in variants 2-3 use a real `bool` for the same idea, which removes
    # the string comparison entirely.
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# with_structured_output binds the schema as a forced tool call and parses the
# reply back into a GradeDocuments instance - no output parsing, no regex.
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# This prompt is used to grade the relevance of a retrieved document to a user question. 
# It instructs the model to give a binary score of 'yes' or 'no' based on whether the document contains keywords or semantic meaning related to the question.

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
# The retrieval_grader is a pipeline that combines the grade_prompt with the structured_llm_grader. 
# It takes a retrieved document and a user question as input, and outputs a binary s
#   core indicating whether the document is relevant to the question.