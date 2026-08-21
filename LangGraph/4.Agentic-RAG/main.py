from dotenv import load_dotenv
from typing import Any, cast

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Running LangGraph/4.Agentic-RAG/main.py")
    print(app.invoke(input=cast(Any, {"question": "What is agent memory?"})))