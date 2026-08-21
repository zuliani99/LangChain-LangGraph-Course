"""
LangGraph lesson 3 / chains.py - the "actor": first responder and revisor.

Both are the same prompt template with a different `first_instruction`, bound to
a different forced tool (AnswerQuestion vs ReviseAnswer). Forcing the tool with
tool_choice is what guarantees structured, critique-bearing output every turn.

NOTE: two defects in this file are documented in the lesson README - the
relative import just below, and the construction of `first_responder`.
"""

import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# BUG: relative import in a module that is executed as a top-level script
# (main.py does `from chains import ...`), which fails with
#     ImportError: attempted relative import with no known parent package
# Fix: `from schemas import AnswerQuestion, ReviseAnswer`.
from .schemas import AnswerQuestion, ReviseAnswer

# gpt-4-turbo: reflexion needs a model strong enough to critique itself usefully.
llm = ChatOpenAI(model="gpt-4-turbo")
# Raw parser: tool calls -> list of dicts, keeping the call id (needed to pair a
# tool result back to its request).
parser = JsonOutputToolsParser(return_id=True)
# Typed parser: tool calls -> validated AnswerQuestion instances.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion]) 
# takes the answer from the llm, create an AnswerQuestion object, that we can easily work with

# The actor prompt. {first_instruction} is the single slot that turns this same
# template into either the drafter or the revisor.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        # A trailing system message: last-position instructions are the least
        # likely to be forgotten in a long transcript.
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    # A CALLABLE partial: re-evaluated at every invoke(), so the timestamp is
    # always current instead of frozen at import time.
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Answer the user's question above in ~250 words.",
)

# BUG: format_prompt() RENDERS a prompt, it does not bind tools - and rendering
# without the required `messages` variable raises KeyError: 'messages' at import
# time. `tools=` / `tool_choice=` are silently treated as template variables.
# The intended construction is the same shape as `revisor` below:
#     first_responder = first_responder_prompt_template | llm.bind_tools(
#         tools=[AnswerQuestion], tool_choice="AnswerQuestion"
#     )
# See "Known issues" in this lesson's README.md.
first_responder = first_responder_prompt_template.format_prompt(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


# Revision policy: ADD what the critique said was missing, CUT what it said was
# superfluous, and cite everything - while holding the 250-word budget.
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# tool_choice forces THIS tool: the model cannot reply in prose, it must return
# a ReviseAnswer payload (answer + reflection + queries + references).
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions,
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomus soc problem domain, "
        " list startups that do that and raised capital"
    )
    # Smoke test for the actor only, no graph involved. Note it builds the chain
    # correctly here (template | bound llm | parser) - unlike `first_responder`.
    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)