"""
Agentic RAG / Adaptive RAG - GENERATE node.

Plain RAG generation: stitch the surviving documents into one context string
and answer from it. Every quality judgement happens elsewhere - before it in
grade_documents, and (variants 2-3) after it in the hallucination/answer
graders that sit on the conditional edge leaving this node.
"""

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Flatten Documents (and any bare strings) into the single {context} slot the
    # prompt expects. Blank line between chunks so the model sees them as
    # separate sources rather than one run-on passage.
    context = "\n\n".join(
        d if isinstance(d, str) else d.page_content for d in documents
    )
    # StrOutputParser is already on the chain, so this returns a plain str.
    generation = generation_chain.invoke({"context": context, "question": question})
    # `documents` is echoed because the downstream hallucination grader
    # (variants 2-3) needs the very facts this answer was built from.
    return {"documents": documents, "question": question, "generation": generation}