"""
Agentic RAG / Adaptive RAG - WEB_SEARCH node.

The fallback when the local corpus is judged insufficient. Tavily results are
collapsed into ONE synthetic Document and APPENDED to whatever survived
grading, so the generator sees local knowledge and fresh web knowledge in the
same context.
"""

from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

from dotenv import load_dotenv
load_dotenv()

# Built at import time, so TAVILY_API_KEY must already be in the environment -
# hence the load_dotenv() call directly above.
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- RUNNING WEB SEARCH ---")
    question = state["question"]
    # .get() rather than [] because in Adaptive RAG the router can make this the
    # ENTRY node, in which case no retrieve step has run and the key is absent.
    documents = state.get("documents")

    # The raw question is used verbatim as the search query - no rewriting. That
    # is the cheap option; a query-rewriting node here is the natural upgrade.
    tavily_results = web_search_tool.invoke({"query": question})
    # Only the "content" field survives; URLs are discarded, so the final answer
    # cannot cite its web sources (contrast with LangChain lesson 6).
    joined_tavily_result = "\n".join(
        [result["content"] for result in tavily_results["results"]]
    )

    # All three hits become a SINGLE Document. That matters: on a later pass the
    # relevance grader will judge this blob all-or-nothing rather than per-result.
    web_results = Document(
        page_content=joined_tavily_result,
    )
    # Append when local documents survived grading, otherwise start the list.
    if documents:
        cast(list[Document], documents).append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

# Standalone smoke test for this node alone, with a hand-built state. Handy for
# checking the Tavily key without paying for a whole graph run.
if __name__ == "__main__":
    print("Running LangGraph/4.Agentic-RAG/graph/nodes/web_search.py")
    web_search(
        state={
            "question": "agent_memory",
            "documents": [],
            "generation": "",
            "web_search": False,
        }
    )