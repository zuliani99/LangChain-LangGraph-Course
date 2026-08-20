from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()  # take environment variables from .env.

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


MAX_ITERATIONS = 10 # maximum number of iterations the agent will run before stopping
MODEL = "gpt-4o-mini" # model name to use for the agent

# --- Tools (LangChain Tools @tool decoration) ---


@tool
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

@tool
def apply_discount(price: float, discount: str) -> float:
    """Apply a discount tier to a price and return the discounted price.
        Available discount tiers:
        - GOLD 20% for prices above $500
        - SILVER 10% for prices between $100 and $500
        - BRONZE 5% for prices below $100
    """
    print(f"    >> Executing apply_discount with price: {price}, discount: {discount}")
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
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=MODEL, model_provider="openai", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

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

        ai_message = llm_with_tools.invoke(messages)

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

        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

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