

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoining, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> bool:
    """
    Determines whether the agent should continue reasoning or not.
    
    :param state: The current message state containing the conversation history.
    :return: True if the agent should continue, False otherwise.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT
   

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoining)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(
    AGENT_REASON, should_continue, {
        END: END,
        ACT: ACT
    }
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(
    output_file_path="LangGraph/1.React-Agent-Function-Calling/graph.png"
)


if __name__ == "__main__":
    print("Hello, LangGraph!")
    res = app.invoke({"messages": [HumanMessage(
        content="What is the weather in Tokyo? List it and then triple it"
    )]})
    print(res["messages"][LAST].content)