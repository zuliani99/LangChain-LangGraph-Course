
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chain import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    # The messages in the graph, which are used to generate and reflect on tweets.
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph) -> MessageGraph:
    """Generates a tweet based on the messages in the state."""
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    """Reflects on the generated tweet and provides critique and recommendations."""
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> bool:
    """Determines whether to continue the reflection process based on the number of messages."""
    if len(state["messages"]) > 6:
        return END
    return REFLECT



if __name__ == "__main__":
    print("Starting Reflection Agent...")

    builder = StateGraph(state_schema=MessageGraph)
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)

    builder.add_conditional_edges(GENERATE, should_continue, path_map={
        # with some conditions, we can either end the process or continue reflecting on the generated tweet
        END: END,
        REFLECT: REFLECT
    })
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

    response = graph.invoke(input)