"""
LangGraph lesson 1 / react.py - the agent's "equipment": model + tools.

Kept in its own module so nodes.py can import the bound model without circular
imports. Nothing runs here; this file only builds objects.
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# A deterministic tool the model cannot fake: it proves the graph really looped
# back through the tool node instead of doing the arithmetic in its head.
@tool
def triple(num: float) -> float:
    """
    param: num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


# One search tool + one computation tool, so a single question ("weather in
# Tokyo, then triple it") forces TWO sequential tool calls through the graph.
tools = [TavilySearch(max_results=1), triple]

# bind_tools attaches the JSON schemas to every request. temperature=0.0 because
# tool-argument generation should be deterministic, never creative.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(tools)