"""
LangGraph lesson 2 / chain.py - the two personas of the reflection loop.

Same model, two system prompts:
    generate_chain -> writes (and rewrites) the tweet
    reflect_chain  -> plays the critic and grades it
Both read the SAME message list, which is the trick that makes reflection work:
the critic sees the draft, the writer then sees the critique.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# The CRITIC. Asking for concrete, actionable notes (length, virality, style)
# matters: "make it better" produces vague feedback the writer cannot act on.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        # Injects the whole conversation so far after the system message; the
        # graph state is passed in under this exact key.
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            # This clause is what turns a one-shot writer into a reviser: on the
            # second pass the critique is in the history, so it must be applied.
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# One shared model instance - the personas differ only by prompt, not by weights.
llm = ChatOpenAI(model="gpt-4o-mini")

# Minimal LCEL chains: dict -> PromptValue -> AIMessage.
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm