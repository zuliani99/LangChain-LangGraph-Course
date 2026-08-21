"""
Lesson 5 / query - Retrieval-Augmented Generation, three ways.

Requires ingestion.py to have been run first (the Pinecone index must exist).

Implementation 0: no RAG            -> the model answers from parametric memory
Implementation 1: manual RAG        -> retrieve / format / prompt / invoke by hand
Implementation 2: RAG with LCEL     -> the same four steps as one composed Runnable

Run from the repository root:
    uv run python LangChain/5.rag-gist/main.py
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter



load_dotenv()

print("Initializing components...")

# Same embedding model as ingestion.py - non negotiable, see ingestion.py.
embeddings = OpenAIEmbeddings()
# temperature=0 for RAG: we want the answer grounded in the context, not invented.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Connects to the EXISTING index (ingestion.py filled it); nothing is written here.
vectorStore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

# as_retriever() wraps the store in a Runnable: str -> List[Document].
# k=3 -> the 3 nearest chunks by cosine similarity. Raising k improves recall
# but costs prompt tokens and can bury the relevant chunk in noise.
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# "based only on the following context" is the anti-hallucination clause: it
# tells the model to abstain rather than fall back on its own knowledge.
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:

    {context}

    Question: {question}
    
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def retreivial_chain_without_lcel(query: str):
    """
    Simple retrieval cain without LCEL.
    Manually retrives documents, gormats them and generates a response using the LLM.
    
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: embed the query and pull the k nearest chunks from Pinecone.
    # Srep 1: retreive relevant documents
    docs = retriever.invoke(query)

    # Step 2: format the documents into a string for the prompt
    context = format_docs(docs)

    # Step 3: fill the template -> a PromptValue (a list of chat messages).
    # Step 3: format the prompt with the context and the question
    messages = prompt_template.format_prompt(context=context, question=query)

    # Step 4: one LLM call. Note every step is manual, sequential and sync-only.
    # Step 4: invoke the LLM with the formatted prompt
    response = llm.invoke(messages)

    return response.content


##########################################################################################
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
##########################################################################################
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operatior (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.batch() for multiple inputs
    - Type safety: better intergation with LangChain's tyle system
    - Less code: more coincise and readable
    - Reusable: chain can be saved, shared and composed with other chains
    - Better debugging: LangChain prvides better observability tools
    """
    # RunnablePassthrough.assign() forwards the input dict unchanged AND adds the
    # computed keys to it: {"question": q} -> {"question": q, "context": "..."}.
    # That is why prompt_template can then read both placeholders.
    #
    # BUG: `itemgetter[str]("question")` raises
    #      TypeError: type 'operator.itemgetter' is not subscriptable.
    #      operator.itemgetter is not a generic - the subscript must be dropped:
    #          context=itemgetter("question") | retriever | format_docs
    #      See "Known issues" in this lesson's README.md.
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter[str]("question") | retriever | format_docs
        )
        #| format_docs # langchain automatically convert format_docs to a Runnable lambdas
        # itemgetter -> retriever -> format_docs is itself a sub-chain: plain
        # callables are auto-wrapped as RunnableLambda when piped.
        | prompt_template     # dict            -> PromptValue
        | llm                 # PromptValue     -> AIMessage
        | StrOutputParser()   # AIMessage       -> str
    )
    return retrieval_chain
    


if __name__ == "__main__":
    print("Rretrieving documents...")

    query = "What is Pinecone in machine learning?"

    # ===============================================================
    # Option 0: Raw invocation without RAG
    # ===============================================================
   
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (no RAG)")
    print("=" * 70)
    # Baseline: no retrieval. Compare this answer with the two below to see what
    # grounding actually buys you (facts, freshness, citable source text).
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content) 


    # ===============================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ===============================================================
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retreivial_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel) 


    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    # Same result as implementation 1, but this object also supports .stream(),
    # .batch(), .ainvoke() and can be piped into another chain as-is.
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)