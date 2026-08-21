"""
Agentic RAG / Corrective RAG (CRAG) - the generation chain.

The only chain in the package that produces prose; everything else emits a
structured verdict. Kept deliberately minimal so the surrounding graph, not the
prompt, is what improves answer quality.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# NOTE: no `model=` here, so langchain-openai falls back to its default
# (gpt-3.5-turbo) - variants 2 and 3 pin gpt-4o-mini for the same chain. Worth
# aligning before comparing answer quality across the three graphs.
llm = ChatOpenAI(temperature=0)

# Inline equivalent of hub.pull("rlm/rag-prompt") — the hub module was removed
# from the langchain package, and pulling public prompts now requires a
# LangSmith API key.
# Three constraints doing real work: answer only from {context}, admit ignorance
# instead of inventing, and stay under three sentences. The "just say that you
# don't know" clause is what the hallucination grader (variants 2-3) then checks.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

# StrOutputParser unwraps AIMessage -> str, so the GENERATE node can drop the
# result straight into state["generation"].
generation_chain = prompt | llm | StrOutputParser()