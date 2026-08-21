"""
Agentic RAG / Self-RAG - graph assembly.

Everything CRAG does, plus the graph now grades its OWN answer before letting it
out:

    ... -> generate -+-(not supported)-> generate      (hallucinated: retry)
                     +-(not useful)----> websearch     (grounded but off-target)
                     +-(useful)--------> END

Two sequential gates on the edge leaving GENERATE:
    1. hallucination_grader - is the answer supported by the documents?
    2. answer_grader        - does it actually address the question?
Order matters: an ungrounded answer is rejected before anyone asks if it is useful.

Compiled at IMPORT time, and importing this module also writes graph.png.
"""

from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

# The two self-assessment chains that variant 1 does not have.
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
# Star imports are what let the wiring below read as bare names (RETRIEVE,
# retrieve, ...). Convenient here, but it does hide where each name comes from.
from graph.consts import *
from graph.nodes import *

from graph.state import GraphState

load_dotenv()


# --- Router: read the flag grade_documents set, pick the next node. -----------
# Routers never mutate the state; they return the NAME of the next node.
def decide_to_generate(state):
    print("--- ASSESS GRADED DOCUMENTS ---")

    # web_search is True as soon as ONE retrieved chunk was graded irrelevant.
    if state["web_search"]:
        print(
            "--- DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEB_SEARCH
    else:
        print("--- DECISION: GENERATE ---")
        return GENERATE


# --- Router: the self-critique gate on the way out of GENERATE. ---------------
# Returns one of three literals, mapped to nodes in the path map further down.
# Unlike decide_to_generate (which reads a flag) this router makes up to TWO
# extra LLM calls per pass, so it is the expensive part of the graph.
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # Gate 1 - groundedness. The walrus keeps the grade around for debugging;
    # the branch itself only needs the truthiness.
    if hallucination_grade := score.binary_score: # type: ignore
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        # Gate 2 - usefulness. Only reached once the answer is known to be
        # grounded, so a failure here means "right facts, wrong question".
        if answer_grade := score.binary_score: # type: ignore
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            # Grounded but off-target -> regenerating from the same context
            # would not help, so widen the evidence with a web search instead.
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        # Hallucinated -> retry generation with the SAME documents. Note there
        # is no attempt counter: a model that keeps hallucinating loops until
        # LangGraph's recursion limit (default 25) raises.
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


    
workflow = StateGraph(GraphState)

# Four nodes, shared by all three variants. What differs between the variants is
# only the EDGES - the routing logic, not the work each node does.
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
# The corrective branch: good documents -> answer; any bad document -> top up
# from the web first.
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)

# The self-correction loop. Two of the three outcomes send the graph BACKWARDS,
# which is why this cannot be expressed as a plain LCEL chain.
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,   # hallucinated -> try again
        "useful": END,               # grounded and on-target -> done
        "not useful": WEB_SEARCH,    # grounded but off-target -> get more evidence
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)
# REDUNDANT: the conditional edge above already routes GENERATE -> END on
# "useful". Verified harmless with the pinned langgraph (the retry loop still
# iterates and the extra edge is folded into the branch map), but it is dead
# wiring - delete it to keep graph.png honest.
workflow.add_edge(GENERATE, END)

# compile() validates the topology and returns a Runnable.
app = workflow.compile()

# NOTE: absolute, machine-specific path - it only works on this checkout. A path
# built from __file__ (or draw_mermaid() for offline text) would be portable.
# Also a network call: the renderer is mermaid.ink.
app.get_graph().draw_mermaid_png(
    output_file_path="/Users/zulle/Github Project/LangChain-LangGraph-Course/LangGraph/4.Agentic-RAG/2.self_rag/graph.png"
)