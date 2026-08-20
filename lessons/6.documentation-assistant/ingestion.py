import asyncio
import os
import ssl
from typing import Any, Dict, List


import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (
    Colors, log_info, 
    log_success, log_warning, 
    log_error, log_header
)

load_dotenv()  # Load environment variables from .env file


# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10 # removing it the error 429 appears, so we set it to 10 seconds to avoid hitting the rate limit too quickly.
)

#vectorStore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorStore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)

'''
The difference between Chroma and Pinecone is that Chroma is a local vector store that stores embeddings on disk,
while Pinecone is a managed vector database service that provides scalable and efficient storage and retrieval of embeddings in the cloud.
'''

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breath=20, max_pages=1000)
tavily_crawl = TavilyCrawl()



async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl dicumentation from https://python.langchain.com/", 
        color=Colors.PURPLE
    )

    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/", # the URL of the documentation site to crawl
        "max_depth": 5, # limit the depth of crawling to 5 levels deep
        "extract_depth": "advanced", # treat more data as relevant for extraction, including code snippets, tables, and structured data
        "instructions": "content on ai agents" # this is a hint to the model to focus on content related to AI agents, which is the main topic of interest
    })

    # all_docs = res["results"]
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
        if result.get("raw_content")
    ]
    # all_docs contains the crawled documents, each represented as a Document object with its content and source URL.

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} documents from the documentation site.",
    )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap.", color=Colors.PURPLE)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Successfully split {len(all_docs)} documents into {len(splitted_docs)} chunks.")

    # Process documents in batches asynchronously
    await index_documents_async(splitted_docs, batch_size=500) # depending on the size of the documents, you may want to adjust the batch size for optimal performance.

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")




async def index_documents_async(documents: List[Document], batch_size: int = 50) -> None:
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} dicuments to vector store",
        color=Colors.DARKCYAN
    )

    # Create bathces
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of size {batch_size} for processing.")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_number: int) -> None:
        try:
            await vectorStore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_number + 1}/{len(batches)} to vector store.")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_number + 1}/{len(batches)}. Error: {str(e)}")

    # Process batches concurrently
    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful_batches = sum(1 for result in results if not isinstance(result, Exception))

    if successful_batches == len(batches):
        log_success(
            f"VectorStore Indexing: Successfully added all batches to vector store."
        )
    else:
        log_warning(
            f"VectorStore Indexing: Added {successful_batches}/{len(batches)} batches successfully. Some batches failed."
        )

if __name__ == "__main__":
    asyncio.run(main())