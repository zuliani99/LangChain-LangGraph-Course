"""
LangGraph lesson 1 / nodes.py - the two nodes of the ReAct graph.

A LangGraph node is just a function `state -> partial state`. It returns only
the keys it wants to update; LangGraph merges the result into the state using
each key's reducer (for MessagesState["messages"] the reducer APPENDS).
"""

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions."""

def run_agent_reasoining(state: MessagesState) -> MessagesState:
    """
    This function runs the agent reasoning process using the provided state.
    It utilizes the llm and tools defined in react.py to generate a response.
    
    :param state: The current message state containing the conversation history.
    :return: The agent's response as a string.
    """
    # Here you would implement the logic to process the state and generate a response
    # using the llm and tools. This is a placeholder for demonstration purposes.
    
    # Example of how you might use the llm to generate a response
    # The system message is prepended on every call instead of being stored in
    # the state, so it never gets duplicated as the transcript grows.
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    
    # Return a PARTIAL state. `add_messages` appends this AIMessage to the
    # existing list - returning the full list here would duplicate the history.
    return {"messages": [response]}


# ToolNode is the prebuilt executor: it reads .tool_calls off the last AIMessage,
# runs each tool (in parallel when there are several), and appends one
# ToolMessage per call with the matching tool_call_id. This is the ~20 lines of
# dispatch code written by hand in LangChain lesson 4.
tool_node = ToolNode(tools=tools)