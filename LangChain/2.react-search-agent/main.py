"""
Lesson 2 - ReAct search agent with `create_agent`.

Same agent, two tool backends:
    main_tools()  -> a fake local tool defined with @tool
    main_tavily() -> the real TavilySearch web-search tool

`create_agent` builds a prebuilt LangGraph ReAct loop (model -> tools -> model)
so you never write the while-loop yourself. Lesson 4 rebuilds that loop by hand.

Run from the repository root:
    uv run python LangChain/2.react-search-agent/main.py
"""

from dotenv import load_dotenv

# Must run before the model/tool objects are constructed: they read
# OPENAI_API_KEY / TAVILY_API_KEY from the environment at __init__ time.
load_dotenv()  # take environment variables from .env.

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langchain_tavily import TavilySearch


# @tool turns a plain Python function into a LangChain Tool. The tool NAME comes
# from the function name, the DESCRIPTION from the docstring and the JSON SCHEMA
# from the type hints - all three are sent to the model so it can decide when to
# call it. A vague docstring is the #1 cause of an agent ignoring a tool.
@tool
def search(query: str) -> str:
    """
    A search tool that takes a query and returns the search results.
    """
    # Here you would implement the actual search logic, for example using an API or a database.
    # For demonstration purposes, we'll just return a mock response.
    return f"Search results for '{query}'"

def main_tools() -> None:
    print("Hello from the React Search Agent lesson!")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0.5
    )  # create an ChatOpenAI instance of the LLM with the model name and temperature

    tools = [search]

    # create_agent compiles a ReAct graph: the model is bound to the tools, and
    # every tool call is routed to a ToolNode whose result is appended to the
    # message list before the model is called again - until it stops calling tools.
    agent = create_agent(model=llm, tools=tools)
    # The agent state is a message list; invoke() returns the FULL transcript,
    # not just the answer: Human -> AI(tool_calls) -> Tool -> AI(final).
    response = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    # Read the final answer with: response["messages"][-1].content
    print(response)



def main_tavily() -> None:
    print("Hello from the React Search Agent lesson!")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0.5
    )  # create an ChatOpenAI instance of the LLM with the model name and temperature

    # TavilySearch is a ready-made tool (name/description/schema already defined)
    # backed by a search API tuned for LLMs: it returns clean text, not raw HTML.
    # Requires TAVILY_API_KEY in .env.
    tools = [TavilySearch()]

    # Identical agent code - only the tool list changed. Tools are pluggable.
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    print(response)

if __name__ == "__main__":
    # Toggle these two lines to compare a stub tool with a real web search:
    # the fake tool answers "Search results for '...'", so the model has to fall
    # back on its own knowledge, while Tavily returns real, citable sources.
    main_tools()
    #main_tavily()