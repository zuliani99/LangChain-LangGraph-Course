# =============================================================================
# Lesson 6 / backend - the RAG "brain" behind the Streamlit UI.
#
# run_llm(query) -> {"answer": str, "context": List[Document]}
#
# Instead of a fixed retrieve->prompt->answer chain (lesson 5), retrieval is
# exposed to the model as a TOOL. The agent therefore decides whether to search,
# what to search for, and may search several times before answering - and it can
# skip retrieval entirely for a greeting.
#
# The block quoted out immediately below is the earlier, more heavily annotated
# draft of this same module (k=5, explicit temperature). The live implementation
# starts after it. Both are kept side by side for the lesson.
# =============================================================================

'''import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load environment variables from .env file

# Initialize the OpenAI embeddings model, same as in ingestion.py, to ensure consistency across the application.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorStore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

# Initialize chat model
model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.0,
    model_provider="openai"
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Dict[str, Any]:
    """Retrieve relevant documentation to help answer user question about LangChain."""
    # Retreive top 5 relevant documents from the vector store based on the query
    retrieved_docs = vectorStore.as_retriever(search_kwargs={"k": 5}).invoke(query)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # This is done to provide the model with context from the retrieved documents, 
    # including their source URLs and content.

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so." # -> to not allow hallucinations
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    # Iterate through the messages to find any ToolMessage that contains artifacts (the retrieved documents)
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"): # search for artifacts in the messages, which are the retrieved documents from the vector store
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
                # append the retrieved documents to the context_docs list for later reference in the output
    
    return {
        "answer": answer, # llm response to the user query
        "context": context_docs # list of retrieved documents that were used to generate the answer
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)



    '''



import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# MUST match ingestion.py exactly: query vectors and stored vectors have to come
# from the same model or similarity search is meaningless.
# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Read-only handle on the index that ingestion.py populated.
#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
# Provider-agnostic factory: change the string to switch model or vendor.
# Initialize chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai")


# response_format="content_and_artifact" makes the tool return a 2-tuple:
#   [0] content  -> the string the MODEL sees (must be text)
#   [1] artifact -> an arbitrary Python object the model never sees, stored on
#                   the ToolMessage for the application to use
# That is how the raw Document objects survive the round-trip and reach the UI,
# instead of being flattened into the prompt string.
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # NOTE: k belongs in the retriever config, not in invoke(). Passing it here
    # is ignored, so this actually returns the retriever default (k=4 anyway).
    # The explicit form is: vectorstore.as_retriever(search_kwargs={"k": 4})
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    
    # The source URL is embedded in the text the model reads, which is what makes
    # the "always cite the sources" instruction in the system prompt satisfiable.
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        # Explicit permission to fail: the single most effective anti-hallucination
        # instruction in a RAG system prompt.
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    # A fresh agent per call. In production build it once at module level - this
    # rebuilds the graph on every request.
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # No history is passed: this app is stateless, every question stands alone.
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # The loop has ended, so the last message is the AI's plain-text answer.
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Walk the transcript and collect every retrieved Document. There may be
    # several ToolMessages if the agent searched more than once.
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)