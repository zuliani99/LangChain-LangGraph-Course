# Lesson 4 — Agents Under the Hood

> `create_agent` deleted. The same shopping agent is rebuilt three times, each version stripping away one more layer of abstraction, until nothing is left but a `for` loop, a regex and a prompt.

| | |
|---|---|
| **Entry points** | [`1_agent_loop_langchain_tool_calling.py`](1_agent_loop_langchain_tool_calling.py) · [`2_agent_loop_raw_function_calling.py`](2_agent_loop_raw_function_calling.py) · [`3_raw_react_promt.py`](3_raw_react_promt.py) |
| **Concepts** | The agent loop, `bind_tools`, JSON tool schemas, `tool_call_id`, the ReAct prompt, stop tokens, scratchpads, LangSmith tracing |
| **Requires** | Step 1: `OPENAI_API_KEY`. Steps 2–3: a running local **Ollama**. All: `LANGSMITH_API_KEY` (optional) |

---

## 1. The thesis

**An agent is a `while` loop around an LLM.** That is the entire idea. Everything else — `@tool`, `bind_tools`, `ToolNode`, `create_agent` — is ergonomics layered on top of these five steps:

```
1. send the conversation to the model
2. did it ask for a tool?      no  → that is the answer, stop
3.                             yes → run the tool
4. append the result to the conversation
5. goto 1                      (with a hard iteration cap)
```

All three scripts implement exactly those five steps. What changes is **who does the plumbing.**

| | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| Framework | LangChain | none (raw Ollama) | none |
| Model | `gpt-4o-mini` | local Ollama | local Ollama |
| Tool definition | `@tool` decorator | plain function | plain function |
| Tool schema | auto-generated | **hand-written JSON** | **prose in the prompt** |
| Model knows about tools? | yes (native API) | yes (native API) | **no** |
| Call format | `AIMessage.tool_calls` | `message.tool_calls` | regex over raw text |
| Result plumbing | `ToolMessage` + `tool_call_id` | `{"role": "tool", ...}` | string concatenation |
| Memory | list of Message objects | list of dicts | one growing string |
| Robustness | high | high | **fragile** |

## 2. The shared scenario

A shopping assistant with two tools — `get_product_price(product)` and `apply_discount(price, tier)` — and a system prompt full of **STRICT RULES**:

1. never guess a price, always call the tool;
2. only apply a discount *after* getting a real price;
3. never do the arithmetic yourself;
4. if no tier is specified, ask.

This is not decoration. It is the experiment: the tools are the only source of truth, so any number in the final answer that did not come out of a tool is a hallucination you can spot immediately.

> Prices differ on purpose between step 3 (`laptop: 1299.99`) and steps 1–2 (`laptop: 999.99`). If a run reports the wrong one, the model answered from memory instead of calling the tool.

---

## 3. Step 1 — LangChain tool calling

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog and return the price as a float."""
```

### The loop

```python
llm = init_chat_model(model=MODEL, model_provider="openai", temperature=0)
llm_with_tools = llm.bind_tools(tools)

for iteration in range(1, MAX_ITERATIONS + 1):
    ai_message = llm_with_tools.invoke(messages)
    tool_calls = ai_message.tool_calls

    if not tool_calls:                      # ← termination: a plain-text answer
        return ai_message.content

    tool_call = tool_calls[0]               # one tool per iteration, deliberately
    observation = tools_dict[tool_call["name"]].invoke(tool_call["args"])

    messages.extend([
        ai_message,
        ToolMessage(content=observation, tool_call_id=tool_call["id"]),
    ])
```

Points worth internalising:

- **`init_chat_model`** is the provider-agnostic factory. Change `model_provider` to `"anthropic"` or `"ollama"` and this file is unchanged.
- **`bind_tools` does not execute anything.** It returns a *new* runnable that attaches the tool schemas to every request. Running the tools is your job — that is the whole point of the lesson.
- **`ai_message.tool_calls` is normalised.** Every provider has a different wire format; LangChain flattens them all into `[{"name":..., "args":{...}, "id":...}]`.
- **Both halves must be appended, in order:** the `AIMessage` that requested the tool, then the `ToolMessage` carrying the result. The `tool_call_id` is what pairs them — omit it and the provider rejects the next request.
- **`tool_calls[0]` only.** Models can request several tools in one turn; taking just the first serialises the agent. Simpler to read, but it also means an unanswered `tool_call` stays in the history — real code loops over all of them.
- **`MAX_ITERATIONS` is not optional.** Without it, a model that keeps calling tools bills you until you notice.

### Tool result typing

`observation` is a `float`; `ToolMessage(content=...)` coerces it to `"49.99"`. Everything a model reads is text — always.

### Run it

```bash
uv run python "LangChain/4.agents-under-the-hood/1_agent_loop_langchain_tool_calling.py"
```

Expected shape:

```
--- Iteration 1---
    [Tool Selected] get_product_price with args: {'product': 'keyboard'}
    >> Executing get_product_price with product: keyboard
    [Tool Result] 49.99
--- Iteration 2---
    [Tool Selected] apply_discount with args: {'price': 49.99, 'discount': 'gold'}
    [Tool Result] 39.992
--- Iteration 3---
Final Answer: The keyboard costs $49.99; with the gold tier discount it is $39.99.
```

---

## 4. Step 2 — Raw function calling, no LangChain

Same loop, LangChain removed. Three concrete differences, marked as `Difference N` in the source:

### Difference 1 — no `@tool`

Plain functions. `@traceable` only adds LangSmith observability; it does not make anything callable by the model.

### Difference 2 — hand-written JSON Schema

```python
tools_for_llm = [{
    "type": "function",
    "function": {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog ...",
        "parameters": {
            "type": "object",
            "properties": {"product": {"type": "string", "description": "..."}},
            "required": ["product"],
        },
    },
}]
```

**This is precisely what `@tool` generated for you in step 1.** Writing it once by hand is the fastest way to understand what a "tool" actually is on the wire: a name, a description and a JSON Schema. Note the maintenance hazard — rename the Python function and this JSON silently goes stale.

### Difference 3 — no `Message` classes, no `tool_call_id`

Messages are plain dicts, and Ollama pairs a result to its request by **`tool_name`**, not by id:

```python
messages.extend([
    ai_message,
    {"role": "tool", "content": str(observation), "tool_name": tool_name},
])
```

Every provider differs here. Normalising that difference away is a large part of what LangChain is for.

### Bonus — self-correction via errors

`apply_discount` in this version *validates eligibility* (GOLD only above $500) and raises:

```python
try:
    observation = tool_to_use(**tool_args)
except ValueError as e:
    observation = f"Error: {e}"     # ← fed back to the model as the observation
```

Handing the error to the model instead of crashing is a genuinely important agent pattern: the model reads the message, understands *why* it failed, and picks a valid tier or explains the problem to the user. Ask for a gold discount on a `$49.99` keyboard and watch it happen.

### Run it

```bash
ollama serve                      # in another terminal
ollama pull <a tool-calling model>
uv run python "LangChain/4.agents-under-the-hood/2_agent_loop_raw_function_calling.py"
```

> **`MODEL = "gemma4"`** — check this tag against your local `ollama list`. It must be a model you have pulled **and** one that supports tool calling; a model without tool support will simply answer in prose and the loop will exit on iteration 1.

---

## 5. Step 3 — Raw ReAct prompting: no tool API at all

This is how agents worked **before** providers shipped function calling. The model is a plain text completer that has never heard of tools. All agency is emulated:

### 5.1 Tool descriptions built by introspection

```python
original_function = getattr(tool_function, "__wrapped__", tool_function)
signature = inspect.signature(original_function)
docstring  = inspect.getdoc(tool_function) or ""
descriptions.append(f"{tool_name}{signature} - {docstring}")
```

`__wrapped__` unwraps the `@traceable` decorator; without it you would show the model the *decorator's* signature (`*, config=None`) instead of the real one.

### 5.2 The ReAct prompt

```
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_product_price, apply_discount]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

The model is **imitating a worked example**, not honouring an API contract. Nothing forces it to comply.

### 5.3 The stop token

```python
options={"stop": ["\nObservation"], "temperature": 0}
```

Without it the model happily writes its *own* `Observation:` line — inventing the tool result. The stop token freezes generation at exactly the point where reality must be injected. **This one line is what makes text-based agents possible at all.**

### 5.4 Parsing

```python
final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
action_match       = re.search(r"Action:\s*(.+)", output)
action_input_match = re.search(r"Action Input:\s*(.+)", output)
```

Termination is now a *pattern match on English*, not the factual `if not tool_calls` of step 1. Arguments are split positionally on commas and stay strings — which is why `apply_discount` has to `float()` its own input, and why a comma inside an argument value breaks everything.

### 5.5 The scratchpad

```python
scratchpad += f"{output}\nObservation: {observation}\nThought:"
full_prompt = prompt + scratchpad
```

Memory is one growing string, re-sent whole on every iteration. The trailing `"Thought:"` is a nudge that pushes the model straight back into the format.

### Run it

```bash
uv run python "LangChain/4.agents-under-the-hood/3_raw_react_promt.py"
```

Run it several times. You will eventually see it fail — a missing `Action Input:`, a stray sentence before `Thought:`, arguments in the wrong order. **That fragility is the lesson.** Native tool calling exists because this approach breaks under load.

---

## 6. LangSmith tracing

Every script is instrumented with `@traceable`:

```python
@traceable(name="LangChain Agent Loop", description="An agent that can call tools ...")
@traceable(run_type="tool")
@traceable(name="Ollama Chat", run_type="llm")
```

Note that step 2 has **no LangChain objects at all** yet still traces perfectly: observability is independent of the framework. Enable it with:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=agents-under-the-hood
```

The trace tree shows every iteration, every tool call and its arguments, latency and token counts per step — the fastest way to debug why an agent looped or picked the wrong tool.

## 7. Gotchas

- **`MODEL = "gemma4"`** in steps 2–3 must match a tag you have actually pulled.
- **The system prompt contains typos** (`helpul`, `acces`, `dicount`, `proce`, `oes not specigy`, `whoch`). Models tolerate them, but this is a prompt: keep it clean, and treat a rules block as code you review.
- **Prices and tiers differ between step 3 and steps 1–2** — intentionally, as a hallucination detector.
- **Only the first tool call is handled** in steps 1–2. Ask something that needs two independent lookups and observe the dropped call.
- **`MAX_ITERATIONS = 10`** is a cost ceiling, not a nicety.

## 8. Exercises

1. Handle **all** `tool_calls` in one iteration instead of `tool_calls[0]`, appending one `ToolMessage` per call.
2. Delete rule 3 ("never calculate discounts yourself") from the system prompt and see whether the model starts doing the arithmetic in its head.
3. Add the eligibility check from step 2 to step 1's `apply_discount` and implement the same error-feedback pattern.
4. In step 3, make the model output an unparseable action on purpose and add a recovery branch that reminds it of the format instead of `break`.
5. Port step 1 to `model_provider="ollama"` via `init_chat_model` — the loop should not change by one line.
6. Log `len(str(messages))` per iteration and watch the context grow; that is why long agent runs get expensive.

---

**Previous:** [Lesson 3 — Structured search agent](../3.search-agent/README.md) · **Next:** [Lesson 5 — RAG, the gist](../5.rag-gist/README.md)
