"""
Agentic RAG / Adaptive RAG - the question router.

What makes this variant "adaptive": before anything is retrieved, an LLM decides
WHICH source can answer the question at all. On-topic questions enter the RAG
pipeline; everything else skips straight to web search, saving a pointless
retrieval plus one grading call per chunk.
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # Literal makes the two options part of the JSON schema, so the model cannot
    # return a third value. "websearch" is spelled to match consts.WEB_SEARCH,
    # which is what lets route_question() compare the two directly.
    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., description="Given a user question choose to route it to web search or vectorstore."
    )


# gpt-4o rather than the gpt-4o-mini used by the graders: this single call
# decides the entire shape of the run, so a mistake here is the expensive kind.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)


# The prompt must DESCRIBE THE CORPUS - the model has no other way of knowing
# what the vectorstore contains. Change the ingested URLs and this sentence has
# to change with them, or the router starts misrouting.
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Invoked from route_question() in graph.py, wired as the graph's CONDITIONAL
# ENTRY POINT - it runs before any node.
question_router = route_prompt | structured_llm_router