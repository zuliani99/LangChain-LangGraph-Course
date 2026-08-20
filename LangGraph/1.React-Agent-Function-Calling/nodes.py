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
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)