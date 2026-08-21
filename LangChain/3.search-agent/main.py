"""
Lesson 3 - Search agent with a STRUCTURED response.

Extends lesson 2 with `response_format=AgentResponse`: instead of free-form text
the agent must emit an object that validates against a Pydantic schema, so the
result can be consumed by code (a UI, a DB insert, another chain) without regex.

Run from the repository root:
    uv run python LangChain/3.search-agent/main.py
"""

from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# load_dotenv() sits between the imports on purpose: the module-level `llm` and
# `TavilySearch()` below are built at import time and need the keys already set.
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# Every Field(description=...) is serialised into the JSON Schema handed to the
# model. The descriptions are prompt engineering, not documentation: they are the
# only instructions the model gets about how to fill each field.
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    # Nested models are supported: this becomes an array of objects in the schema.
    # default_factory=list keeps the field optional so a source-less answer still
    # validates instead of raising.
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  # create an ChatOpenAI instance of the LLM with the model name and temperature
tools = [TavilySearch()]
# response_format adds a final "structured output" step to the ReAct loop: once
# the model stops calling tools it is forced to fill AgentResponse. The parsed
# object is returned under result["structured_response"], already validated.
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    # A multi-step request: the agent has to search, read the results and only
    # then summarise them into the schema - several ReAct iterations, one call.
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    # result["messages"]            -> full transcript (tool calls included)
    # result["structured_response"]  -> the AgentResponse instance you actually want
    print(result)


if __name__ == "__main__":
    main()