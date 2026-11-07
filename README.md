# Ola Domain Support Agent (CrewAI)

**Completed track: Ola (Business Operations / Customer Support).**

This repository contains a deterministic, MOCK_LLM customer support capstone covering seeded ticket data, a policy knowledge base, two RAG indexes, CrewAI tools and agents, conversation memory, guardrails, FastAPI, evaluation, and governance.

## Dataset design

The generator exports the deterministic `SUPPORT_TICKETS` list using **seed 42**, **50 tickets**, category weights **Billing 0.25, Technical Issue 0.20, Account Access 0.20, Product Defect 0.15, General Inquiry 0.20**, status weights **Open 0.15, In Progress 0.25, Escalated 0.10, Resolved 0.30, Closed 0.20**, and an escalated probability of **0.20**. Resolution times range from **0.5 to 72 hours** (quick fixes through multi-day investigations); ticket ages are integers from **0 to 30 days**. The deterministic dataset report records 50 tickets, every category at least three times, every status present, and **20% escalated**.

## RAG and calibration

`kb/` contains the twelve required policy topics. `rag_core.py` indexes 500-character fixed chunks with 100-character overlap and sentence-based chunks, embeds each locally with SentenceTransformers, and stores them in separate ChromaDB cosine collections.

`config/threshold.json` records calibration from five in-scope and two out-of-scope queries. The measured in-scope top-1 cosine scores are **0.4530, 0.6897, 0.7155, 0.6460, 0.5662** (mean **0.6141**). The out-of-scope scores are **0.1095, 0.0389** (mean **0.0742**). The configured threshold is **0.2812**, midway between the lowest in-scope score and highest out-of-scope score, between the observed clusters. Run `python scripts/calibrate.py` to refresh the transcript and threshold for the local model.

`transcripts/chunk_eval.txt` compares document-level precision and recall across both collections. Current reported averages are fixed-size **P=0.1339, R=1.0000, F1=0.2335** and sentence-based **P=0.2252, R=1.0000, F1=0.3633**; sentence-based chunking is the recommendation. `scripts/check_kb.py` checks topic and sentence coverage.

## Agent, memory, guardrails, and API

- The Retrieval Agent owns `rag_lookup`; only the Lookup Agent owns `check_support_ticket_status`; the Composer has no tools. The score formula is `0.6 × escalated + 0.4 × ((30 − days_since_created) / 30)`, clamped to [0, 1]. A threshold of 0.65 identifies escalated tickets created within the last 26 days.
- CrewAI uses a `BaseLLM` MOCK_LLM and CrewAI's built-in ReAct executor flow. The mock reads tool schemas from `from_agent.tools`, returns tool actions in CrewAI's text ReAct format, and composes the prior grounded answer and ticket summary for the final draft.
- `memory.py` uses LangChain `InMemoryChatMessageHistory` with `RunnableWithMessageHistory`; it is process-local.
- Guardrails mask fixed-format phone numbers, detect prompt injection, and refuse low-similarity answers. Use fabricated names, addresses, and payment data; those free-text values are outside the phone masker scope.
- FastAPI exposes `POST /ask`, `POST /add-document`, health/cache endpoints, and `WS /ws/chat`. Request logs are JSON Lines with trace IDs, timings, and masked query text. The request cap is 4,000 estimated tokens.
- The AutoGen review uses a two-agent `RoundRobinGroupChat` with `max_turns=2`, structured `VerdictModel` output, and a local deterministic replay client. The demo shows both an unchanged approval and a revision of an injected refund promise.
- `evaluation_judge.py` scores 15 queries for Accuracy, Grounding, Completeness, and Safety under MOCK_LLM. `governance.py` demonstrates least autonomy, Medium risk classification, and oversized-request rejection.

## Setup and run

Python 3.10 or newer is required. Install dependencies in a virtual environment and ensure the SentenceTransformers model is available locally for offline execution.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts\generate_dataset.py
.venv\Scripts\python.exe scripts\check_kb.py
.venv\Scripts\python.exe scripts\calibrate.py
.venv\Scripts\python.exe -m ola_support_agent.evaluation
.venv\Scripts\python.exe scripts\grounded_demo.py
.venv\Scripts\python.exe scripts\crew_demo.py
.venv\Scripts\python.exe -m ola_support_agent.memory
.venv\Scripts\python.exe -m ola_support_agent.guardrails
.venv\Scripts\python.exe -m ola_support_agent.autogen_review
.venv\Scripts\python.exe -m ola_support_agent.governance
.venv\Scripts\python.exe -m ola_support_agent.evaluation_judge
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn ola_support_agent.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before CrewAI kickoff; the code sets both before importing CrewAI. MOCK_LLM and AutoGen replay make no language-model API requests.
