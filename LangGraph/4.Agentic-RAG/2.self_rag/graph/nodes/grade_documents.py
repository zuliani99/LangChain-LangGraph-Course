"""
Agentic RAG / Self-RAG - GRADE_DOCUMENTS node.

The heart of Corrective RAG: every retrieved chunk is graded yes/no for
relevance by an LLM, irrelevant ones are dropped, and a single "no" raises the
`web_search` flag so the graph knows the local corpus was not enough.

Cost note: one LLM call PER DOCUMENT. With the default top-4 retriever that is
4 extra calls on every question.
"""

from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False   # stays False only if EVERY document is graded relevant
    for d in documents:
        # Defensive unwrapping: retrieve() yields Documents, but web_search()
        # may have appended one too, and GraphState annotates the list as str.
        if isinstance(d, str):
            document_text = d
        else:
            document_text = getattr(d, "page_content", str(d))
        score = retrieval_grader.invoke(
            {"question": question, "document": document_text}
        )
        # with_structured_output normally returns a GradeDocuments instance, but
        # some model/provider combinations hand back a plain dict - handle both.
        grade = (
            score["binary_score"]
            if isinstance(score, dict)
            else score.binary_score
        )
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # One irrelevant chunk is enough to trigger the fallback. Strict by
            # design: the assumption is that a gap in the corpus is better filled
            # from the web than papered over by the remaining chunks.
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}