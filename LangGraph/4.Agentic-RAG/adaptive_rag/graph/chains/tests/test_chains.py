from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from typing import cast
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"document": doc_text, "question": question}),
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"document": doc_text, "question": "how to make pizza"}),
    )
    
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    generation = generation_chain.invoke({"question": question, "context": doc_text})
    pprint(generation)
