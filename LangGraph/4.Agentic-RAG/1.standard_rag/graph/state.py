"""
Agentic RAG / Corrective RAG (CRAG) - the shared graph state.

Unlike the message-list states of LangGraph lessons 1-3, this is a plain
domain state: no reducers, so every node OVERWRITES the keys it returns. A node
that wants to preserve a value must return it unchanged - which is exactly why
retrieve/grade/generate all echo `question` back in their return dicts.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    A TypedDict representing the state of a graph.

    Attributes:
        question: question asked by the user
        genertion: LLM generation result for the question
        web_search: whether to perform a web search for additional information
        documents: list of documents retrieved from the web search    
    """
    
    question: str      # the user's question, echoed by every node that touches it
    generation: str     # the LLM answer; only set once `generate` has run
    web_search: bool    # flag set by grade_documents -> read by decide_to_generate
    # NOTE: annotated List[str] but it actually holds langchain Document objects
    # (retrieve returns Documents, web_search appends a Document). The nodes work
    # around the mismatch with isinstance(d, str) checks; List[Document] would be
    # the honest annotation.
    documents: List[str]