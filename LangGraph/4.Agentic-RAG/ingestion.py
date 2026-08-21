from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "langgraph_agentic_rag"
PERSIST_DIRECTORY = "./.chroma_db"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

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

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore.add_documents(doc_splits)

retriever = vectorstore.as_retriever()