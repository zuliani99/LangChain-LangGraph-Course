"""
Lesson 4 / step 1 - The agent loop, written by hand with LangChain tool calling.

`create_agent` (lessons 2-3) is replaced by an explicit for-loop so you can see
exactly what an agent is: call the model -> if it asked for a tool, run it and
append the result -> call the model again -> stop when it answers in plain text.

Abstraction level here: HIGH. @tool builds the JSON schema, .bind_tools() sends
it, and the provider returns parsed `tool_calls`. Steps 2 and 3 peel that away.

Run from the repository root:
    uv run python "LangChain/4.agents-under-the-hood/1_agent_loop_langchain_tool_calling.py"
"""

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()  # take environment variables from .env.

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# Hard stop: without it a model that keeps calling tools loops (and bills) forever.
MAX_ITERATIONS = 10 # maximum number of iterations the agent will run before stopping
MODEL = "gpt-4o-mini" # model name to use for the agent

# --- Tools (LangChain Tools @tool decoration) ---


# Difference 1: @tool derives the tool contract automatically ->
#   name        = "get_product_price"
#   description = the docstring below
#   parameters  = {"product": {"type": "string"}} from the type hints
# In step 2 we will have to write that JSON by hand.
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog and return the price as a float."""
    print(f"    >> Executing get_product_price with product: {product}")
    # Stand-in for the database/API call a real pricing tool would make.
    catalog = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "keyboard": 49.99,
    }
    return catalog.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount: str) -> float:
    """Apply a discount tier to a price and return the discounted price.
        Available discount tiers:
        - GOLD 20% for prices above $500
        - SILVER 10% for prices between $100 and $500
        - BRONZE 5% for prices below $100
    """
    print(f"    >> Executing apply_discount with price: {price}, discount: {discount}")
    # NOTE: no eligibility check here - any tier applies to any price. Step 2
    # adds validation and feeds the error back to the model instead of crashing.
    discount_percentages = {
        "gold": 0.20,
        "silver": 0.10,
        "bronze": 0.05,
    }
    discount_percentage = discount_percentages.get(discount.lower(), 0.0)
    discounted_price = price * (1 - discount_percentage)
    return discounted_price


# --- Agent Loop (LangChain Agent) ---

@traceable(name="LangChain Agent Loop", description="An agent that can call tools to answer questions.")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    # Name -> Tool lookup: the model replies with a tool NAME, we need the object.
    tools_dict = {tool.name: tool for tool in tools}
    # init_chat_model is the provider-agnostic factory: swap model_provider to
    # "anthropic"/"ollama" and nothing else in this file changes.
    llm = init_chat_model(model=MODEL, model_provider="openai", temperature=0)
    # bind_tools returns a NEW runnable that ships the tool schemas with every
    # request. It does NOT execute anything - running tools is our job below.
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # `messages` is the agent's entire memory. Every iteration re-sends the whole
    # list, which is why token cost grows quadratically with loop length.
    messages = [
        SystemMessage(
            content=(
                "You are a helpul shopping assistant. "
                "You have acces to a product catalog tool "
                "and a dicount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product prices. "
                "You must use the get_product_price tool to look up real prices.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact proce "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user oes not specigy a discout tier,"
                "ask them whoch tier to use -  do NOT assume one"
            ),  
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration }---")

        # One LLM call per iteration. The model sees the transcript so far and
        # decides: answer, or request another tool.
        ai_message = llm_with_tools.invoke(messages)

        # LangChain normalises every provider's format into this list of dicts:
        # [{"name": ..., "args": {...}, "id": ...}]. Empty list == final answer.
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found in tools_dict.")

        # .invoke() on a @tool validates tool_args against the schema first,
        # then runs the function. Invalid args raise before your code executes.
        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        # Both halves must be appended, in this order: the AIMessage that made
        # the request, then the ToolMessage carrying the answer. The tool_call_id
        # is what pairs them - drop it and the provider rejects the next request.
        messages.extend([
            ai_message,
            ToolMessage(content=observation, tool_call_id=tool_call_id)
        ])

    print("Maximum iterations reached without a final answer.")
    return None



def main():
    print("Hello Lanchain Agent (.bind_tools)!")
    print()
    reult = run_agent(
        "What is the price of a keyboard and what is the discounted price if I apply a gold tier discount?"
    )
    print()
    print("Final Result:")
    print(reult)

if __name__ == "__main__":
    main()