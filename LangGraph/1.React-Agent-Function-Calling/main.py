"""
LangGraph lesson 1 - ReAct agent built explicitly as a graph.

    START -> agent_reason -+-(tool calls)----> act -> agent_reason -> ...
                           +-(no tool calls)-> END

Same behaviour as `create_agent`, but every node and edge is declared by hand,
which is what you need as soon as the flow stops being a plain loop.

Run from the repository root (the graph.png path below is relative to it):
    uv run python LangGraph/1.React-Agent-Function-Calling/main.py
"""

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoining, tool_node

load_dotenv()

# Node names are strings used in add_node/add_edge; constants avoid typos that
# would otherwise only surface at compile() time.
AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1  # index of the most recent message in the state


def should_continue(state: MessagesState) -> bool:
    """
    Determines whether the agent should continue reasoning or not.
    
    :param state: The current message state containing the conversation history.
    :return: True if the agent should continue, False otherwise.
    """
    # The router does NOT mutate the state: it inspects the last AIMessage and
    # returns the NAME of the next node. No tool calls -> the model answered.
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT
   

# MessagesState is the built-in schema {"messages": Annotated[list, add_messages]}
# - i.e. a state with a single append-only message list.
flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoining)
flow.set_entry_point(AGENT_REASON)   # equivalent to add_edge(START, AGENT_REASON)
flow.add_node(ACT, tool_node)

# Conditional edge: after AGENT_REASON run should_continue() and jump to the node
# it names. The third argument is the path map (router output -> node), which is
# also what lets LangGraph draw the dashed branches in the diagram.
flow.add_conditional_edges(
    AGENT_REASON, should_continue, {
        END: END,
        ACT: ACT
    }
)

# Unconditional edge back to the model: this single line is what makes it a LOOP.
flow.add_edge(ACT, AGENT_REASON)

# compile() validates the topology (unreachable nodes, missing entry point) and
# returns a Runnable - so the graph supports invoke/stream/batch like any chain.
app = flow.compile()
# Renders the topology to PNG (needs network access: it calls the mermaid.ink
# renderer). Use .draw_mermaid() for offline text output instead.
app.get_graph().draw_mermaid_png(
    output_file_path="LangGraph/1.React-Agent-Function-Calling/graph.png"
)


if __name__ == "__main__":
    print("Hello, LangGraph!")
    # Two tools, in order: search Tokyo weather, then triple the temperature.
    # Watch the loop run twice through agent_reason before it terminates.
    res = app.invoke({"messages": [HumanMessage(
        content="What is the weather in Tokyo? List it and then triple it"
    )]})
    print(res["messages"][LAST].content)