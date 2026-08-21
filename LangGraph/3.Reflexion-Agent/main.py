"""
LangGraph lesson 3 - Reflexion agent (draft -> research -> revise, N times).

    START -> draft -> execute_tools -> revise -+-(<= MAX_ITERATIONS)-> execute_tools
                                               +-(>  MAX_ITERATIONS)-> END

Difference from lesson 2 (Reflection): the critique is no longer just an opinion.
The actor must emit `search_queries` alongside its self-critique, those queries
are actually run against Tavily, and the revision is grounded in the results and
forced to carry numbered citations.

Run from the repository root:
    uv run python LangGraph/3.Reflexion-Agent/main.py
"""

import os
from dotenv import load_dotenv

from typing import Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()

# Number of research+revision rounds. Each round costs 1 LLM call + up to 3
# Tavily searches, so raising it gets expensive fast.
MAX_ITERATIONS = 2

def draft_node(state: MessagesState) -> MessagesState:
    """Draft an initial response."""
    # The hasattr() guard is a workaround for the broken `first_responder` in
    # chains.py: a PromptValue has no .invoke, so this falls through to
    # `first_responder | revisor`, which a PromptValue also cannot do. Fix the
    # chain in chains.py and this whole expression collapses to
    #     first_responder.invoke({"messages": state["messages"]})
    response = (first_responder if hasattr(first_responder, "invoke") else first_responder | revisor).invoke(  # type: ignore[attr-defined]
        {"messages": state["messages"]}
    )
    return {"messages": [response]}


def revise_node(state: MessagesState) -> MessagesState:
    """Revise the response based on critique."""
    # The revisor sees the ENTIRE transcript: original question, previous draft,
    # its own critique and the raw search results appended by execute_tools.
    response = revisor.invoke({"messages": state["messages"]})  # type: ignore[attr-defined]
    return {"messages": [response]}


def event_loop(state: MessagesState):
    """Loop through drafting and revising until the answer is satisfactory or max iterations reached."""
    # Counting ToolMessages is how the loop measures progress: one search round
    # == one ToolMessage appended by execute_tools. No quality gate is involved,
    # the agent simply stops after a fixed research budget.
    count_tol_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tol_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"



# Three nodes, one cycle. Only the edge out of "revise" is conditional; the
# draft -> tools -> revise path always runs at least once.
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
# Path map given as a LIST here (vs a dict in lessons 1-2): allowed when the
# router already returns the exact node names.
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

# Text mermaid instead of draw_mermaid_png(): no network call, prints to stdout.
print(graph.get_graph().draw_mermaid())



# NOTE: module level, no `if __name__ == "__main__"` guard - importing this file
# fires a full (paid) agent run as a side effect.
res = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
            )
        ]
    }
)
'''
res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)
'''


# The final message is a forced ReviseAnswer TOOL CALL, not prose: the answer
# lives in the call arguments, which is why it is dug out by hand here.
# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)