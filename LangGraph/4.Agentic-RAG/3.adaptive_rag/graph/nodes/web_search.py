from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- RUNNING WEB SEARCH ---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
        [result["content"] for result in tavily_results["results"]]
    )

    web_results = Document(
        page_content=joined_tavily_result,
    )
    if documents:
        cast(list[Document], documents).append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

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