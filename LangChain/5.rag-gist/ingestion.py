"""
Lesson 5 / ingestion - Build the vector index (run this ONCE, before main.py).

The four canonical RAG ingestion steps:
    LOAD    mediumblog1.txt      -> Document
    SPLIT   Document             -> ~1000-char chunks
    EMBED   chunk text           -> 1536-dim vectors (text-embedding-ada-002)
    STORE   vectors + text       -> Pinecone index (INDEX_NAME)

Run from the repository root:
    uv run python LangChain/5.rag-gist/ingestion.py
"""

import os
from dotenv import load_dotenv

#from langchain_community.document_loaders import TextLoader ====> DEORECATED
# UnstructuredLoader handles ~60 formats (pdf, html, docx, ...) behind one API,
# so the pipeline below stays identical when the source file type changes.
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    # Get the environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Debug print kept from the lesson - NEVER do this outside a local sandbox,
    # it writes live secrets to stdout and into any log collector.
    # Print the environment variables to verify they are loaded correctly
    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
    print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
    print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")

    print("Ingesting...")
    # Path built from __file__ so the script works from any working directory.
    file_path = os.path.join(os.path.dirname(__file__), "mediumblog1.txt")
    # max_characters is deliberately huge: we want ONE Document out of the loader
    # and let CharacterTextSplitter below own the chunking policy.
    loader = UnstructuredLoader(file_path, chunking_strategy="basic", max_characters=1000000, encoding="utf-8")
    document = loader.load()  # -> List[Document(page_content=..., metadata=...)]

    print("Splitting...")
    # Chunk size is the core RAG trade-off: too small loses context, too large
    # dilutes the embedding and wastes prompt tokens at query time.
    # chunk_overlap=0 is the simple case; overlap (100-200) avoids cutting a
    # sentence that answers the question exactly across two chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks.")

    # The embedding model is part of the index contract: main.py MUST query with
    # the same model, or the query vector lands in a different geometry space.
    embeddings = OpenAIEmbeddings()

    print("Creating vector store...")
    # from_documents = embed every chunk + upsert into an EXISTING Pinecone index.
    # Create the index first in the Pinecone console with dimension 1536 and
    # metric "cosine". Re-running this script appends duplicates, it does not reset.
    vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Vector store created successfully.")