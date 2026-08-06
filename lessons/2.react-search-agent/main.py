from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langchain_tavily import TavilySearch


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

    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    print(response)



def main_tavily() -> None:
    print("Hello from the React Search Agent lesson!")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0.5
    )  # create an ChatOpenAI instance of the LLM with the model name and temperature

    tools = [TavilySearch()]

    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    print(response)

if __name__ == "__main__":
    main_tools()
    #main_tavily()