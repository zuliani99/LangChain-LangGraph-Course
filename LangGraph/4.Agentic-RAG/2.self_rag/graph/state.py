from typing import List, TypedDict


class GraphState(TypedDict):
    """
    A TypedDict representing the state of a graph.

    Attributes:
        question: question asked by the user
        genertion: LLM generation result for the question
        web_search: whether to perform a web search for additional information
        documents: list of documents retrieved from the web search    
    """
    
    question: str
    generation: str
    web_search: bool
    documents: List[str]