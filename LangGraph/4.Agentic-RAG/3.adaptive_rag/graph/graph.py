"""
Agentic RAG / Adaptive RAG - graph assembly.

Everything Self-RAG does, plus a routing decision taken BEFORE any work happens:

    START -+-(on-topic)--> retrieve -> grade_documents -> ...
           +-(off-topic)-> websearch -> generate -> ...

The router reads the question and picks the datasource. An off-topic question
("how to make pizza") skips retrieval entirely, which saves one retrieval plus
one grading call per chunk - the whole point of "adaptive".

Three layers of quality control now stack up:
    route_question   - which source can answer this at all?      (before)
    grade_documents  - is each retrieved chunk relevant?         (during)
    grade_generation - grounded? useful?                         (after)

Compiled at IMPORT time, and importing this module also writes graph.png.
"""

from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

# The two self-assessment chains from variant 2...
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
# ...plus the datasource router, which is what variant 3 adds.
from graph.chains.router import question_router, RouteQuery
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


# --- Router: the conditional ENTRY POINT, runs before any node. ---------------
def route_question(state: GraphState) -> str:
    print("--- ROUTING QUESTION ---")
    question = state["question"]
    source_result = question_router.invoke({"question": question})
    # with_structured_output usually returns the model instance directly, but
    # some provider paths hand back a dict - model_validate normalises both.
    source: RouteQuery = (
        source_result
        if isinstance(source_result, RouteQuery)
        else RouteQuery.model_validate(source_result)
    )
    # Works because consts.WEB_SEARCH == "websearch" == the schema Literal.
    if source.datasource == WEB_SEARCH:
        print("--- ROUTE QUESTION TO WEB SEARCH ---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
        print("--- ROUTE QUESTION TO RAG ---")
        return RETRIEVE
    else:
        # Unreachable while the schema stays a two-value Literal - kept as a
        # guard for when a third datasource is added.
        raise ValueError(f"Unknown datasource: {source.datasource}")

    
    
workflow = StateGraph(GraphState)

# The same four nodes as variants 1 and 2 - only the edges differ.
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

# Replaces set_entry_point(RETRIEVE): the graph now decides where to START.
# This is the one structural change between Self-RAG and Adaptive RAG, and it is
# also why web_search.py had to switch to state.get("documents") - entering here
# means no retrieve step has populated that key.
workflow.set_conditional_entry_point(
    route_question,
    {
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE,
    },
)


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
    output_file_path="/Users/zulle/Github Project/LangChain-LangGraph-Course/LangGraph/4.Agentic-RAG/3.adaptive_rag/graph.png"
)