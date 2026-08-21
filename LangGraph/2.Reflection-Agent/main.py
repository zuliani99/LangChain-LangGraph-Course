"""
LangGraph lesson 2 - Reflection agent (generate <-> critique loop).

    START -> generate -+-(<= 6 messages)-> reflect -> generate -> ...
                       +-(>  6 messages)-> END

The critique is re-injected as a HumanMessage so the writer treats it as user
feedback rather than as something it said itself. No tools, no external data:
the only thing improving the output is the model reading its own draft.

Run from the repository root (graph.png path is relative to it):
    uv run python LangGraph/2.Reflection-Agent/main.py
"""

from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chain import generate_chain, reflect_chain

# A hand-written state schema, equivalent to the built-in MessagesState used in
# lesson 1. Annotated[..., add_messages] is the REDUCER: it tells LangGraph to
# append what a node returns instead of overwriting the list.
class MessageGraph(TypedDict):
    # The messages in the graph, which are used to generate and reflect on tweets.
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph) -> MessageGraph:
    """Generates a tweet based on the messages in the state."""
    # The AIMessage is appended as-is, so the next reflect step can read it.
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    """Reflects on the generated tweet and provides critique and recommendations."""
    res = reflect_chain.invoke({"messages": state["messages"]})
    # KEY DETAIL: the critique is re-labelled as a HumanMessage. From the
    # generator's point of view the feedback then arrives from the user, which
    # models follow far more reliably than their own previous assistant turns.
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> bool:
    """Determines whether to continue the reflection process based on the number of messages."""
    # Termination is a message COUNT, not a quality judgement: 1 human + 3
    # generate/reflect pairs -> stop after roughly 3 revision rounds. A real
    # system would ask a grader model whether another pass is still worth it.
    if len(state["messages"]) > 6:
        return END
    return REFLECT



if __name__ == "__main__":
    print("Starting Reflection Agent...")

    builder = StateGraph(state_schema=MessageGraph)
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)  # always write a first draft before critiquing

    builder.add_conditional_edges(GENERATE, should_continue, path_map={
        # with some conditions, we can either end the process or continue reflecting on the generated tweet
        END: END,
        REFLECT: REFLECT
    })
    # Closes the cycle: every critique is followed by a rewrite.
    builder.add_edge(REFLECT, GENERATE)
    graph = builder.compile()

    graph.get_graph().draw_mermaid_png(
        output_file_path="LangGraph/2.Reflection-Agent/graph.png"
    )

    input = HumanMessage(
        content="""
        Make this tweet better:
        @LangChainAI
        - newly Tool Calling feature is seriously underrated.
        After a ling wait, it's  here- making the implementatio of agents across differentmodels with function calling
        made a video covering their newest blog post 
        """
    )

    # invoke() only returns the final state. Use graph.stream(input) to watch
    # each draft and each critique as they are produced.
    response = graph.invoke(input)