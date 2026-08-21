from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--- RETRIEVE NODE ---")

    question = state["question"]

    document = retriever.invoke(question)
    return {
        "documents": document,
        "question": question
    }