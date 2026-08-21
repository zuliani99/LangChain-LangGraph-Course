"""
Lesson 1 - Hello World: the smallest possible LangChain program.

Demonstrates the three building blocks every LangChain app is made of:
    PromptTemplate  ->  ChatModel  ->  (response)
wired together with the LCEL pipe operator (`|`) into a single Runnable.

Run from the repository root:
    uv run python LangChain/1.hello-world/main.py
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

# Reads the .env file at the project root and injects its keys into os.environ.
# ChatOpenAI never receives the key explicitly: it picks OPENAI_API_KEY up from
# the environment at construction time, so load_dotenv() must run BEFORE it.
load_dotenv()  # take environment variables from .env.


def main():
    print("Hello from langchain-course!")

    # The "context" we want the model to work on. In a real app this string
    # would come from a scraper, a database row or a RAG retriever.
    information = """
    Elon Reeve Musk[1] (AFI:[ˈiːlɒn ˈɹiːv ˈmʌsk]) (Pretoria, 28 giugno 1971) è un imprenditore e politico sudafricano con cittadinanza canadese naturalizzato statunitense.

    Ricopre i ruoli di fondatore, amministratore delegato e direttore tecnico della compagnia aerospaziale SpaceX,[2] fondatore di The Boring Company[3] e della società di intelligenza artificiale xAI, cofondatore di Neuralink e OpenAI,[4] amministratore delegato e product architect della multinazionale automobilistica Tesla,[5] proprietario e presidente di X (precedentemente Twitter)[6], nonché cofondatore di PayPal attraverso la fusione di X.com e Confinity[7]. Ha inoltre proposto un sistema di trasporto superveloce conosciuto come Hyperloop One, posta in liquidazione il 21 dicembre 2023.[8] Tramite SpaceX gestisce Starlink, una costellazione satellitare per la fornitura di Internet ad alta velocità e bassa latenza a tutto il pianeta.[9]

    Secondo Forbes, al 12 giugno 2026, con un patrimonio stimato di 1100 miliardi di dollari,[10] risultava essere la persona più ricca del mondo[11] e la persona più ricca della storia contemporanea, avendo raggiunto per la prima volta la cifra di 1000 miliardi di dollari [12]. Tale status è durato pochi giorni ed era legato alla sopravvalutazione della azioni SpaceX in seguito alla procedura di IPO[13].

    In virtù dei risultati ottenuti nei settori dell’ingegneria e della tecnologia, in particolare attraverso lo sviluppo di aziende come Tesla e SpaceX, è stato descritto da diversi media internazionali come una delle figure imprenditoriali più influenti e innovative dell’età contemporanea[14], ed è stato incluso dalla rivista Time tra le persone più influenti al mondo[7][14][15].

    Dal 20 gennaio al 29 maggio 2025 è stato a capo del Dipartimento dell'Efficienza Governativa statunitense.[16][17][18]
    """

    # The prompt template: plain text with {placeholders} in single braces.
    # Everything the model is asked to do lives here, not in Python code.
    summary_template =  """
    given the information {information} about the person, I want you to create:
    1. A brief summary of their life and career
    2. A list of their most notable achievements
    """

    # f-string is not used here because we want to keep the template as a string with placeholders for the input variables.

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], # use the variable name that matches the input variable in the template
        template=summary_template,
    )

    # temperature=0.5 -> mildly creative. Use 0.0 when you need deterministic,
    # reproducible output (extraction, classification, tool calling).
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5) # create an ChatOpenAI instance of the LLM with the model name and temperature
    # Swap-in a local model: same interface, zero code changes downstream.
    # This is the whole point of LangChain's ChatModel abstraction.
    #llm = ChatOllama(model="gemma3", temperature=0.5) # create an ChatOllama instance of the LLM with the model name and temperature

    chain = summary_prompt_template | llm # create a chain that first formats the prompt and then sends it to the LLM
    # chain is runnable, so we can invoke it with the input variable "information"

    # invoke() runs the chain synchronously: dict -> PromptValue -> AIMessage.
    # The same object also exposes .stream(), .batch(), .ainvoke() for free.
    response = chain.invoke(input={"information": information})
    # AIMessage.content holds the text; the object also carries .response_metadata
    # (token usage, finish reason, model name) which is useful for cost tracking.
    print(response.content) # print the response from the LLM

if __name__ == "__main__":
    main()
