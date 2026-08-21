"""
Agentic RAG / Corrective RAG (CRAG) - node name constants.

Node names are plain strings in LangGraph's API (add_node / add_edge / path
maps). Centralising them here means a typo is a NameError at import time
instead of a silent dead branch discovered at runtime.
"""

# we define the node types here so that we can use them in the graph definition

RETRIEVE = "retrieve" # name of the node type for retrieving documents
GRADE_DOCUMENTS = "grade_documents" # name of the node type for grading documents
GENERATE = "generate" # name of the node type for generating a response based on the retrieved documents and the question
WEB_SEARCH = "websearch" # name of the node type for performing a web search to retrieve documents
# NOTE: the string "websearch" is also one of the two literals the router
# schema can emit (graph/chains/router.py), which is what lets route_question()
# compare source.datasource directly against this constant.
