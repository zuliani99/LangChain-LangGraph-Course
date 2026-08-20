from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()  # take environment variables from .env.

import ollama

MAX_ITERATIONS = 10 # maximum number of iterations the agent will run before stopping
MODEL = "gemma4" # local Ollama model to use for the agent (must support tool calling)

# --- Tools (LangChain Tools @tool decoration) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog and return the price as a float."""
    print(f"    >> Executing get_product_price with product: {product}")
    catalog = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "keyboard": 49.99,
    }
    return catalog.get(product.lower(), 0.0)

@traceable(run_type="tool")
def apply_discount(price: float, discount: str) -> float:
    """Apply a discount tier to a price and return the discounted price.
        Available discount tiers:
        - GOLD 20% for prices above $500
        - SILVER 10% for prices between $100 and $500
        - BRONZE 5% for prices below $100
    """
    print(f"    >> Executing apply_discount with price: {price}, discount: {discount}")
    discount_tiers = {
        "gold": (0.20, lambda p: p > 500),
        "silver": (0.10, lambda p: 100 <= p <= 500),
        "bronze": (0.05, lambda p: p < 100),
    }
    tier = discount.lower()
    if tier not in discount_tiers:
        raise ValueError(f"Unknown discount tier: {discount}")

    discount_percentage, is_eligible = discount_tiers[tier]
    if not is_eligible(price):
        raise ValueError(
            f"The '{discount}' tier does not apply to a price of ${price:.2f}. "
            "GOLD applies above $500, SILVER between $100 and $500, BRONZE below $100."
        )

    discounted_price = price * (1 - discount_percentage)
    return discounted_price



tools_for_llm = [
    {
        "type": "function",
        "function":  {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog and return the price as a float.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up.",
                    },
                },
                "required": ["product"],
            },
        }
    },
    {
        "type": "function",
        "function":  {
            "name": "apply_discount",
            "description": (
                "Apply a discount tier to a price and return the discounted price. "
                "Available discount tiers: GOLD 20% for prices above $500, "
                "SILVER 10% for prices between $100 and $500, "
                "BRONZE 5% for prices below $100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount": {
                        "type": "string",
                        "description": (
                            "The discount tier to apply. "
                            'Valid values are: "gold", "silver", or "bronze".'
                        ),
                    },
                },
                "required": ["price", "discount"],
            },
        }
    }
]
# Difference 2: Without @tool, we must MANUALY define JSON schema for each function.
# This is exactly what LangChain's @ tool decorator generates automatically
# from the function's type hints and docstring. We will use the same schema as LangChain's @tool decorator generates.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    """A wrapper around the ollama.chat function to make it traceable."""
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)


# --- Agent Loop (LangChain Agent) ---

@traceable(name="LangChain Agent Loop", description="An agent that can call tools to answer questions.")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }


    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {"role": "system",
            "content": (
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
        },
        {"role": "user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration }---")

        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found in tools_dict.")

        try:
            observation = tool_to_use(**tool_args)
        except ValueError as e:
            # Feed the validation error back to the model as the tool result
            # instead of crashing, so it can correct itself (e.g. pick a
            # different tier or tell the user why the request is invalid).
            observation = f"Error: {e}"

        print(f"    [Tool Result] {observation}")

        # Difference 3: Ollama has no tool_call_id like LangChain/OpenAI - tool
        # results are matched back to their call via "tool_name" instead.
        messages.extend([
            ai_message,
            {"role": "tool", "content": str(observation), "tool_name": tool_name},
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