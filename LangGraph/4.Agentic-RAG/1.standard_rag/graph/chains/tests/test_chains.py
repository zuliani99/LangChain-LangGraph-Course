"""
Agentic RAG / Corrective RAG (CRAG) - chain-level tests.

These are integration tests, not unit tests: every one of them makes real LLM
and vector-store calls and therefore costs money. The point is not coverage but
CALIBRATION - checking that each grader says "yes" to an obvious positive and
"no" to an obvious negative, which is the only way to know a prompt-based judge
actually discriminates.

BROKEN AS COMMITTED: the import below pulls graph.chains.router, which only
exists in variant 3 (3.adaptive_rag). Collecting this module raises
ModuleNotFoundError, so no test in the file runs. Delete that import line -
nothing in this file uses question_router or RouteQuery.

Run from this variant's folder:
    cd LangGraph/4.Agentic-RAG/<variant> && uv run pytest . -s -v
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from typing import cast
from pprint import pprint

# parents[3] == the variant folder (tests -> chains -> graph -> variant), which
# is what makes `graph.*` and `ingestion` importable when pytest is launched
# from somewhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
# BUG: graph/chains/router.py does not exist in this variant - it is introduced
# in 3.adaptive_rag. This line alone makes the whole module fail to import, and
# neither name is used below. Remove it.
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever

# Positive control: "agent memory" IS covered by the ingested Lilian Weng posts,
# so the top hit must grade as relevant. A failure here means retrieval or the
# grader prompt is broken, and every downstream decision is unreliable.
def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"document": doc_text, "question": question}),
    )

    assert res.binary_score == "yes"


# Negative control, and the subtler of the two: the document is still fetched
# with "agent memory", but graded against an unrelated question. This isolates
# the GRADER - a judge that always says "yes" would pass the test above and fail
# this one.
def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"document": doc_text, "question": "how to make pizza"}),
    )
    
    assert res.binary_score == "no"


# No assertion - just prints the answer for eyeballing. Asserting on free-form
# generated prose is brittle; the graders in variants 2-3 are the programmatic
# way to judge it.
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    generation = generation_chain.invoke({"question": question, "context": doc_text})
    pprint(generation)
