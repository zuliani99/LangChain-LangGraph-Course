"""
Agentic RAG / Corrective RAG (CRAG) - graph assembly.

    START -> retrieve -> grade_documents -+-(all relevant)---> generate -> END
                                          +-(any irrelevant)-> websearch -> generate -> END

The "corrective" idea: never trust similarity search on its own. Grade every
retrieved chunk, drop the bad ones, and if anything was dropped, top the context
up with a live web search before answering.

Compiled at IMPORT time, and importing this module also writes graph.png.
"""

from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

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


workflow = StateGraph(GraphState)
# Four nodes, shared by all three variants. What differs between the variants is
# only the EDGES - the routing logic, not the work each node does.
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

# Always start by retrieving: this variant has no notion of an off-topic
# question. Variant 3 replaces this line with a conditional entry point.
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
# Both paths converge on GENERATE, and generation is final: this variant never
# inspects its own answer. That is exactly what variant 2 adds.
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# compile() validates the topology and returns a Runnable.
app = workflow.compile()

# NOTE: absolute, machine-specific path - it only works on this checkout. A path
# built from __file__ (or draw_mermaid() for offline text) would be portable.
# Also a network call: the renderer is mermaid.ink.
app.get_graph().draw_mermaid_png(output_file_path="/Users/zulle/Github Project/LangChain-LangGraph-Course/LangGraph/4.Agentic-RAG/1.standard_rag/graph.png")