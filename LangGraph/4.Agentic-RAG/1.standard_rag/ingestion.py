"""
Agentic RAG / Corrective RAG (CRAG) - ingestion: build the local Chroma index.

Identical in all three variants, so each one owns a self-contained corpus:
three Lilian Weng blog posts on agents, prompt engineering and adversarial
attacks. That narrow, known corpus is what makes the graders interesting -
"agent memory" is inside it, "how to make pizza" is definitively outside it.

Imported (not executed) by graph/nodes/retrieve.py, which pulls `retriever`
from here. Ingestion therefore happens as an import side effect, guarded so it
only runs the first time.
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "langgraph_agentic_rag"
# Relative path -> the index is created under the CURRENT working directory.
# Run main.py from this variant's folder or you will silently build a second,
# empty .chroma_db somewhere else.
PERSIST_DIRECTORY = "./.chroma_db"

# The corpus. Deliberately small and topically narrow: the router (variant 3)
# and the relevance grader only have a meaningful decision to make when there
# is a clear inside/outside boundary.
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Chroma persists to disk, so the embeddings survive between runs - unlike the
# Pinecone lessons, this costs nothing after the first ingestion and needs no
# cloud account.
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings(),
)

# Importing this module re-runs the block below every time, so only scrape
# and embed once: skip it if the persisted collection is already populated.
if vectorstore._collection.count() == 0:
    docs = [
        UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000, encoding="utf-8"
        ).load()
        for url in urls
    ]
    docs_list = [item for sublist in docs for item in sublist]

    # UnstructuredLoader's chunking metadata (orig_elements, link_texts, ...) is
    # copied onto every split produced below, ballooning each chunk to tens of
    # thousands of characters. Keep only the source URL.
    for doc in docs_list:
        doc.metadata = {"source": doc.metadata.get("url", "")}

    # from_tiktoken_encoder measures chunk_size in TOKENS, not characters, so
    # 250 here is a real budget against the model's context window rather than
    # an approximation. Small chunks keep each vector topically sharp, which is
    # what makes a per-document binary relevance grade meaningful.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    # Embeds every split and writes it to .chroma_db. Runs once; afterwards the
    # count() guard above short-circuits the whole block.
    vectorstore.add_documents(doc_splits)

# The single object the rest of the package imports. Default search: top-4 by
# cosine similarity.
retriever = vectorstore.as_retriever()