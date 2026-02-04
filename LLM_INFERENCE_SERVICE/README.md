# GenAI Inference Service (Decoder-Only, Production-Oriented)

This repository contains a **production-grade GenAI inference service** built using **FastAPI** and a **decoder-only open-source LLM** (e.g. Qwen, Mistral-style).  
The focus is **behavior control, determinism, and safety**, not just “getting text out of a model”.

This is **not a demo**.  
It is a reference implementation for how GenAI APIs should actually be built.

---

## Key Design Goals

- Deterministic, single-turn answers  
- No prompt leakage or instruction echo  
- Explicit control over verbosity  
- Safe token usage (no runaway generation)  
- Clear separation between:
  - model behavior
  - product logic
  - infrastructure

---

## Architecture Overview

```
Client
  ↓
FastAPI (validation, schema)
  ↓
Prompt Builder (system + user contract)
  ↓
Tokenizer
  ↓
Decoder-only LLM (Causal LM)
  ↓
Post-processing (cleanup, truncation)
  ↓
Clean, single-paragraph answer
```

---

## Core Concepts Implemented

### 1. Decoder-Only Models
- Uses `AutoModelForCausalLM`
- Compatible with models like:
  - Qwen
  - Mistral
  - LLaMA-style architectures
- No encoder–decoder assumptions

---

### 2. Single-Turn Completion (Not Chat)

This service **does NOT run a multi-turn chat loop**.

Why:
- Chat-style prompts cause continuation hallucinations
- APIs should return **one answer per request**
- Stateful chat belongs in a higher layer

We intentionally use a **completion-style prompt**.

---

### 3. Prompt Contract (Minimal and Safe)

The prompt is deliberately short to reduce leakage:

```
Answer the following question in a single paragraph of plain text.

Question:
{user_prompt}

Answer:
```

Rules are enforced **in code**, not by begging the model.

---

### 4. Output Sanitization (Non-Negotiable)

The model is treated as an **untrusted text generator**.

Post-processing guarantees:
- No instruction echo
- No system prompt leakage
- No continuation
- No markdown
- Single paragraph only

This is how production systems actually work.

---

### 5. Adaptive Token Budgeting

Instead of a fixed `max_new_tokens`, the system:

1. Classifies the task (`short`, `medium`, `long`)
2. Chooses an expected token range
3. Adds a safety margin
4. Enforces a global hard cap

This improves:
- Latency
- Cost predictability
- Hallucination control

---

### 6. Expected Output Length Enforcement

Output length is enforced **after generation**, not left to the model.

- Expected word ranges per task
- Hard truncation beyond limits
- Deterministic results

Tokens are a **budget**, not a goal.

---

## Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI app
│   ├── model.py         # ModelService (LLM logic)
│   ├── schemas.py       # Request/response validation
│   └── utils.py         # Token counting, sanitization, limits
│
├── requirements.txt
└── README.md
```

---

## ModelService Responsibilities

The `ModelService` class handles:

- Loading tokenizer and model
- Device placement (CPU / GPU)
- Prompt construction
- Text generation
- Output cleanup and enforcement

It **does not**:
- Manage HTTP
- Store state
- Handle user sessions

---

## Example API Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is a large language model?",
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

---

## Example Response

```json
{
  "output": "A large language model is a type of artificial intelligence trained on vast amounts of text data to understand and generate human language. By learning statistical patterns in language, it can perform tasks such as answering questions, summarizing information, generating content, and assisting with reasoning across many domains."
}
```

Guaranteed properties:
- One paragraph
- No markdown
- No instruction echo
- No follow-up questions

---

## Development Workflow

### Run locally (with auto-reload)

```bash
uvicorn app.main:app --reload
```

Note:
- Reload = **process restart**
- Model is reloaded on each change
- This mirrors production behavior

---

## Why This Design Works in Production

- Decoder-only models are probabilistic → code enforces determinism
- Prompts reduce bad behavior → sanitization eliminates it
- Token limits prevent cost explosions
- Output contracts stay stable even if the model changes

You can **swap models** (e.g. Qwen → Mistral)  
without changing the API or product logic.

---

## What This Repo Intentionally Does NOT Do

- Multi-turn chat memory
- Streaming tokens (yet)
- Fine-tuning
- RAG / vector search
- UI / frontend

Those belong in **separate layers**.

---

## Next Steps (Optional Extensions)

- Dockerize for deployment
- Add streaming responses
- Add JSON-schema enforced outputs
- Add RAG with a retriever layer
- Add metrics (token usage, latency)

---

## Final Note

This codebase treats LLMs correctly:

> **Models generate text.  
> Systems decide what is allowed to leave.**

If you understand this repository, you understand real GenAI deployment.
