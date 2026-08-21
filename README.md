# LangChain / LangGraph Course — Building AI Agents 🦜🔗

**Working repository for [Eden Marco's *Develop AI Agents with LangChain & LangGraph*](https://www.udemy.com/course/langchain/?couponCode=JULY-2026) course, rebuilt on LangChain v1 / LangGraph v1.**

Ten self-contained lessons that go from a two-line prompt chain to a RAG pipeline that grades its own retrieval, its own answer, and whether it should have retrieved at all. Every lesson folder holds runnable code, **fully annotated line by line**, plus its own deep-dive `README.md`.

![LangChain Logo](/static/LangChain_OSS%20Lockup_light.png)
![LangGraph Logo](/static/LangGraph_OSS%20Lockup_light.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](.python-version)
[![LangChain](https://img.shields.io/badge/langchain-v1-1C3C3C.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/langgraph-v1-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course%20Coupon%20%2412.99-brightgreen)](https://www.udemy.com/course/langchain/?couponCode=JULY-2026)

---

## 📦 What's in here

Unlike the upstream course repository, **everything lives on `main`** — no branch checkouts, no external repos to clone. Each lesson is a directory:

```
LangChain-LangGraph-Course/
├── LangChain/
│   ├── 1.hello-world/              prompt → model → output
│   ├── 2.react-search-agent/       the ReAct loop via create_agent
│   ├── 3.search-agent/             structured (Pydantic) agent output
│   ├── 4.agents-under-the-hood/    the agent loop rebuilt 3× by hand
│   ├── 5.rag-gist/                 RAG reduced to its essentials
│   └── 6.documentation-assistant/  agentic RAG + Streamlit UI
├── LangGraph/
│   ├── 1.React-Agent-Function-Calling/   ReAct as an explicit graph
│   ├── 2.Reflection-Agent/               generate ⇄ critique cycle
│   ├── 3.Reflexion-Agent/                critique + grounded research
│   └── 4.Agentic-RAG/                    corrective → self → adaptive RAG
│       ├── 1.standard_rag/
│       ├── 2.self_rag/
│       └── 3.adaptive_rag/
├── static/                          logos
├── pyproject.toml                   single uv environment for all lessons
└── uv.lock
```

Every folder contains a `README.md` with: an architecture diagram, a section-by-section walkthrough of the code, the exact run command, expected output, gotchas and exercises.

## 📚 The lessons

### LangChain — chains, tools, agents, RAG

| # | Lesson | What you build | Key concepts | Keys needed |
|---|---|---|---|---|
| 1 | **[Hello World](LangChain/1.hello-world/README.md)** | A summariser in 8 lines | `PromptTemplate`, `ChatModel`, LCEL `\|`, `Runnable`, temperature | `OPENAI` *(or Ollama)* |
| 2 | **[ReAct Search Agent](LangChain/2.react-search-agent/README.md)** | An agent that decides to search | `@tool`, tool schemas, `create_agent`, message transcripts | `OPENAI`, `TAVILY` |
| 3 | **[Structured Search Agent](LangChain/3.search-agent/README.md)** | An agent returning a validated object | Pydantic schemas as prompts, `response_format` | `OPENAI`, `TAVILY` |
| 4 | **[Agents Under The Hood](LangChain/4.agents-under-the-hood/README.md)** | The same agent 3×, peeling off abstraction | The agent loop, `bind_tools`, raw JSON schemas, ReAct prompting, stop tokens | `OPENAI` + **Ollama** |
| 5 | **[RAG, the Gist](LangChain/5.rag-gist/README.md)** | Ask questions about a text file | Embeddings, chunking, vector stores, retrievers, LCEL vs manual | `OPENAI`, `PINECONE` |
| 6 | **[Documentation Assistant](LangChain/6.documentation-assistant/README.md)** | Chat over crawled LangChain docs, with citations | Crawling, async batched indexing, **agentic RAG**, `content_and_artifact`, Streamlit | `OPENAI`, `PINECONE`, `TAVILY` |

### LangGraph — explicit control flow

| # | Lesson | What you build | Key concepts | Keys needed |
|---|---|---|---|---|
| 1 | **[ReAct Agent as a Graph](LangGraph/1.React-Agent-Function-Calling/README.md)** | Lesson 2's agent, node by node | `StateGraph`, `MessagesState`, reducers, conditional edges, cycles | `OPENAI`, `TAVILY` |
| 2 | **[Reflection Agent](LangGraph/2.Reflection-Agent/README.md)** | A writer and a critic in a loop | Custom state schema, `add_messages`, role re-labelling | `OPENAI` |
| 3 | **[Reflexion Agent](LangGraph/3.Reflexion-Agent/README.md)** | Self-critique grounded in live research | Forced tool calls (`tool_choice`), schemas as constraints, citations | `OPENAI`, `TAVILY` |
| 4 | **[Agentic RAG](LangGraph/4.Agentic-RAG/README.md)** | Three RAG graphs: corrective, self, adaptive | LLM-as-a-judge, `with_structured_output`, conditional entry points, self-correction cycles, local Chroma | `OPENAI`, `TAVILY` |

## 🧭 The arc

The order is not arbitrary — each lesson exists to break an assumption made by the previous one:

```
1  Hello World          a chain: you decide the control flow
2  ReAct Agent          the MODEL decides the control flow
3  Structured Agent     ...and its answer is typed, not prose
4  Under The Hood       there was never any magic: it's a for-loop
5  RAG                  give the model knowledge it was never trained on
6  Doc Assistant        let the model decide WHEN and WHAT to retrieve
──────────────────────  LangGraph: when a loop is no longer enough
G1 ReAct as a Graph     the same loop, but every edge is yours to change
G2 Reflection           a cycle that isn't model→tools→model
G3 Reflexion            self-critique that is enforced, then researched
G4 Agentic RAG          stop trusting the retriever, the answer, the premise
```

Lesson 4 is the hinge. Read it before LangGraph: once you have written the agent loop three times by hand, `StateGraph` stops looking like a framework and starts looking like a description of something you already understand.

## ▶️ Getting started

### Prerequisites

- **Python 3.12+** (pinned in [`.python-version`](.python-version))
- **[uv](https://docs.astral.sh/uv/)** — or any package manager that reads `pyproject.toml`. **Not conda.**
- An **OpenAI** API key (every lesson except the lesson 1 bonus and the Ollama halves of lesson 4)
- **[Ollama](https://ollama.com/)** running locally for lesson 4 steps 2–3
- Familiarity with git, virtual environments and environment variables. *This is not a beginner Python course.*

### Install

```bash
git clone git@github.com:zuliani99/LangChain-LangGraph-Course.git
```

```bash
cd LangChain-LangGraph-Course && uv sync
```

One environment covers all ten lessons.

### Configure `.env`

Create a `.env` file **at the repository root** (it is git-ignored):

```dotenv
# Always required
OPENAI_API_KEY=sk-...

# Web search — lessons 2, 3, 6, LangGraph 1 and 3
TAVILY_API_KEY=tvly-...

# Vector store — lessons 5 and 6
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
INDEX_NAME=your-index-name          # lesson 5 only; lesson 6 hard-codes "langchain-doc-index"

# Tracing — optional but strongly recommended, especially for lesson 4
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=langchain-langgraph-course
```

> `load_dotenv()` must run **before** any model or tool object is constructed — they read their keys from `os.environ` at `__init__` time, not at call time. Several lessons place the call between the imports for exactly this reason.

### External services

| Service | Used by | Free tier | Setup |
|---|---|---|---|
| [OpenAI](https://platform.openai.com/) | all | no (pay per token) | `gpt-4o-mini` throughout; **LangGraph** lesson 3 uses `gpt-4-turbo` |
| [Tavily](https://tavily.com/) | 2, 3, 6, G1, G3 | yes | search API built for LLMs |
| [Pinecone](https://www.pinecone.io/) | 5, 6 | yes | **create the index first**: dimension `1536`, metric `cosine` |
| [Chroma](https://www.trychroma.com/) | G4 | yes, local | no account needed — the index is a folder on disk |
| [Ollama](https://ollama.com/) | 4 (steps 2–3) | yes, local | needs a **tool-calling** model |
| [LangSmith](https://smith.langchain.com/) | optional, all | yes | tracing and debugging |

## 🚀 Running a lesson

```bash
uv run python LangChain/1.hello-world/main.py
```

Working directory matters for a few files:

| Lesson | Command | Run from |
|---|---|---|
| 1 | `uv run python LangChain/1.hello-world/main.py` | anywhere |
| 1 (bonus) | `uv run python LangChain/1.hello-world/exercise_groq.py` | anywhere — **no keys, no network** |
| 2 | `uv run python LangChain/2.react-search-agent/main.py` | anywhere |
| 3 | `uv run python LangChain/3.search-agent/main.py` | anywhere |
| 4 | `uv run python "LangChain/4.agents-under-the-hood/1_agent_loop_langchain_tool_calling.py"` *(then 2\_, 3\_)* | anywhere |
| 5 | `uv run python LangChain/5.rag-gist/ingestion.py` **then** `.../main.py` | anywhere |
| 6 | `uv run python LangChain/6.documentation-assistant/ingestion.py` **then** `uv run streamlit run LangChain/6.documentation-assistant/main.py` | anywhere |
| G1 | `uv run python LangGraph/1.React-Agent-Function-Calling/main.py` | **repository root** |
| G2 | `uv run python LangGraph/2.Reflection-Agent/main.py` | **repository root** |
| G3 | `uv run python LangGraph/3.Reflexion-Agent/main.py` | **repository root** — *see known issues* |
| G4 | `uv run python main.py` *(also `uv run pytest . -s -v`)* | **inside the chosen variant folder**, e.g. `LangGraph/4.Agentic-RAG/2.self_rag` |

LangGraph 1 and 2 write `graph.png` to a path relative to the repository root, so they must be launched from there. The three Agentic RAG variants go the other way: each resolves `./.chroma_db` against the working directory, so launch them from **inside** the variant folder or you will silently build a second, empty index somewhere else. Everywhere else, Python puts the script's own directory on `sys.path[0]` (and Streamlit does the same), so sibling and sub-package imports resolve regardless of where you are.

> **Lessons 5 and 6 have a one-off ingestion step.** Run `ingestion.py` once to populate the vector index before running the query side. Lesson 6's crawl is long and costs real money — start with a smaller `max_depth` while experimenting.

## 🐛 Known issues

Reproduced against the versions pinned in [`uv.lock`](uv.lock) (`langchain-core 1.6.0`, `langgraph 1.2.11`, Python 3.12). Each is flagged with a `# BUG:` comment in the source and explained with a fix in the relevant lesson README.

| Where | Symptom | Fix |
|---|---|---|
| [`LangChain/5.rag-gist/main.py:122`](LangChain/5.rag-gist/main.py#L122) | `TypeError: type 'operator.itemgetter' is not subscriptable` — the LCEL implementation never runs | drop the subscript: `itemgetter("question")` |
| [`LangGraph/3.Reflexion-Agent/chains.py`](LangGraph/3.Reflexion-Agent/chains.py) | `ImportError: attempted relative import with no known parent package` | `from schemas import ...` (absolute, as in `tool_executor.py`) |
| [`LangGraph/3.Reflexion-Agent/chains.py`](LangGraph/3.Reflexion-Agent/chains.py) | `KeyError: 'messages'` at import — `format_prompt()` renders a template, it does not bind tools | `first_responder_prompt_template \| llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")` |
| [`pyproject.toml`](pyproject.toml) | `pinecore>=0.0.0` is an unrelated placeholder package (typo for `pinecone`) | harmless — the real client arrives transitively via `langchain-pinecone`; remove the line |
| [`LangGraph/3.Reflexion-Agent/main.py`](LangGraph/3.Reflexion-Agent/main.py) | no `if __name__ == "__main__"` guard — importing the module fires a full, paid run | wrap the bottom of the file |
| [`LangGraph/4.Agentic-RAG/1.standard_rag/…/test_chains.py`](LangGraph/4.Agentic-RAG/1.standard_rag/graph/chains/tests/test_chains.py) | imports `graph.chains.router`, which exists only in variant 3 → `ModuleNotFoundError`, no test in the file runs | delete the unused import |
| [`LangGraph/4.Agentic-RAG/1.standard_rag/…/generation.py`](LangGraph/4.Agentic-RAG/1.standard_rag/graph/chains/generation.py) | `ChatOpenAI(temperature=0)` with no `model=` defaults to **gpt-3.5-turbo**, while variants 2–3 pin `gpt-4o-mini` | add `model="gpt-4o-mini"` |
| all three `LangGraph/4.Agentic-RAG/*/graph/graph.py` | `draw_mermaid_png` uses an absolute, machine-specific path and calls mermaid.ink at import time | build the path from `__file__`, or use `draw_mermaid()` |

Four further Agentic RAG notes (redundant edge, private Chroma API, state annotation, undeclared `pytest`) are listed in [that lesson's README](LangGraph/4.Agentic-RAG/README.md#9--known-issues).

**LangGraph lesson 3 does not run as committed.** Apply the two `chains.py` fixes first.

## 🧰 Stack

| Package | Pinned | Role |
|---|---|---|
| `langchain` | `>=1.3.16` | chains, `@tool`, `create_agent` |
| `langgraph` | `>=1.2.11` | `StateGraph`, `ToolNode`, cycles |
| `langchain-openai` | `>=1.6.0` | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain-ollama` | `>=1.1.0` | local models |
| `langchain-tavily` | `>=0.2.18` | `TavilySearch`, `TavilyCrawl`, `TavilyMap`, `TavilyExtract` |
| `langchain-pinecone` | `>=0.2.13` | managed vector store |
| `langchain-chroma` | `>=1.1.0` | local vector store — the whole of LangGraph lesson 4 |
| `langchain-text-splitters` | `>=1.1.2` | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` |
| `langchain-unstructured` | `>=1.0.1` | multi-format document loading |
| `langsmith` | `>=0.11.1` | `@traceable`, trace inspection |
| `streamlit` | `>=1.62.0` | lesson 6 chat UI |

## 🎯 What you'll be able to do

- Compose prompts, models and parsers with LCEL, and know when *not* to
- Write tools whose docstrings and schemas actually get them called
- Explain — and implement from scratch — what an agent loop is
- Build a RAG pipeline end to end: chunking strategy, embeddings, retrieval, grounded prompting
- Turn retrieval into an agent-controlled action, with citations that survive to the UI
- Express non-linear control flow as a LangGraph state machine: cycles, routers, custom reducers
- Make a model critique itself in a way it cannot game, and ground the revision in fresh evidence
- Use an LLM as a structured judge — of retrieval, of groundedness, of usefulness — and wire those verdicts into control flow
- Trace, debug and cost-account an agent run in LangSmith

## 📄 License & credits

Code released under the [Apache 2.0 License](LICENSE).

Course material and original project designs by **[Eden Marco](https://www.udemy.com/course/langchain/?couponCode=JULY-2026)** ([@EdenMarco177](https://twitter.com/EdenMarco177)). This repository is a personal working copy: restructured onto a single branch, ported to LangChain v1 / LangGraph v1, annotated throughout, and extended with per-lesson documentation.

<div align="center">
<strong>Credits by Eden Marco</strong>
</div>
