"""
LangGraph lesson 3 / tool_executor.py - runs the model's self-generated queries.

The twist: AnswerQuestion / ReviseAnswer are not real tools, they are output
SCHEMAS. But the model emits them as tool calls, so LangGraph will look for a
tool node able to handle a call named "AnswerQuestion". We therefore register
run_queries() twice, once under each schema name, and ignore every field of the
payload except `search_queries`.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)  # 5 hits per query, per revision round


def run_queries(search_queries: list[str], **kargs):
    """Run the generated queries."""
    # **kargs swallows answer/reflection/references: they are part of the tool
    # call payload but irrelevant here, and without it the call would TypeError.
    # .batch() runs the 1-3 queries concurrently instead of one after another.
    return tavily_tool.batch([{"query": query} for query in search_queries])

# Same function exposed under two names, because the draft node emits an
# "AnswerQuestion" call and the revise node emits a "ReviseAnswer" call. ToolNode
# dispatches strictly by name, so both entries are required.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            run_queries,
            name=AnswerQuestion.__name__
        ),
        StructuredTool.from_function(
            run_queries,
            name=ReviseAnswer.__name__
        )
    ]
)