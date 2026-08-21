import os
from dotenv import load_dotenv

from typing import Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2

def draft_node(state: MessagesState) -> MessagesState:
    """Draft an initial response."""
    response = (first_responder if hasattr(first_responder, "invoke") else first_responder | revisor).invoke(  # type: ignore[attr-defined]
        {"messages": state["messages"]}
    )
    return {"messages": [response]}


def revise_node(state: MessagesState) -> MessagesState:
    """Revise the response based on critique."""
    response = revisor.invoke({"messages": state["messages"]})  # type: ignore[attr-defined]
    return {"messages": [response]}


def event_loop(state: MessagesState):
    """Loop through drafting and revising until the answer is satisfactory or max iterations reached."""
    count_tol_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tol_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"



builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())



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


# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)