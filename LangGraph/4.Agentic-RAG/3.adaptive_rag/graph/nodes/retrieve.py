"""
Agentic RAG / Adaptive RAG - RETRIEVE node.

The only node that touches the vector store. Deliberately dumb: it embeds the
question, pulls the nearest chunks and stops. Judging whether those chunks are
any good is the next node's job - that separation is what makes the pipeline
"corrective" instead of blindly trusting similarity search.
"""

from typing import Any, Dict

from graph.state import GraphState
# Importing `retriever` triggers ingestion.py, which builds .chroma_db on the
# first run and reuses it afterwards.
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--- RETRIEVE NODE ---")

    question = state["question"]

    # str -> List[Document], top-4 by cosine similarity. No filtering, no score
    # threshold: similarity alone never tells you whether a chunk is USEFUL.
    document = retriever.invoke(question)
    # `question` is echoed back because GraphState has no reducers - anything a
    # node omits from its return dict keeps its previous value, and echoing it
    # keeps each node's contract explicit.
    return {
        "documents": document,
        "question": question
    }