
import os
from dotenv import load_dotenv

#from langchain_community.document_loaders import TextLoader ====> DEORECATED
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

    # Print the environment variables to verify they are loaded correctly
    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
    print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
    print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")

    print("Ingesting...")
    file_path = os.path.join(os.path.dirname(__file__), "mediumblog1.txt")
    loader = UnstructuredLoader(file_path, chunking_strategy="basic", max_characters=1000000, encoding="utf-8")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    print("Creating vector store...")
    vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Vector store created successfully.")