"""
Agentic RAG / Corrective RAG (CRAG) - entry point.

Run from THIS folder, not the repository root: `graph` and `ingestion` are
imported as top-level modules, and ingestion.py resolves .chroma_db relative to
the working directory.

    cd LangGraph/4.Agentic-RAG/1.standard_rag && uv run python main.py
"""

from dotenv import load_dotenv
from typing import Any, cast

# Must precede the graph import: importing graph.graph transitively builds every
# ChatOpenAI / TavilySearch object, all of which read their keys from os.environ.
load_dotenv()

# Importing `app` compiles the graph AND writes graph.png as a side effect.
from graph.graph import app

if __name__ == "__main__":
    print("Running LangGraph/4.Agentic-RAG/standard_rag/main.py")
    # cast(Any, ...) only silences the type checker: GraphState declares every
    # key as required, but LangGraph is happy to start from a partial state and
    # let the nodes fill in the rest.
    print(app.invoke(input=cast(Any, {"question": "What is agent memory?"})))