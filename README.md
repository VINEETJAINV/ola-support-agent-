# Ola Domain Support Agent (CrewAI)

**Completed track: Ola (Business Operations / Customer Support).**

## Dataset design choices (Part 1, Task 1)

The ticket generator uses seed **42** and **50 records**. Category weights are
Billing **0.25**, Technical Issue **0.20**, Account Access **0.20**, Product
Defect **0.15**, and General Inquiry **0.20**; status weights are Open **0.15**,
In Progress **0.25**, Escalated **0.10**, Resolved **0.30**, and Closed **0.20**.
Escalation is drawn with probability **0.20**. Synthetic ticket amounts are
stored as `amount_inr` and are uniformly generated from **INR 50 to INR 5,000**
using an independent seeded random stream, so adding amounts does not change
the existing ticket categories, statuses, or escalation flags. Resolution
times range from **0.5 to 72 hours** for quick fixes through multi-day
investigations, and ticket ages are integers from **0 to 30 days**.
The generated dataset has **50 records**, satisfies every category/status minimum,
and has **20.0% escalated tickets**; the per-category and per-status counts are in
`transcripts/dataset_validation.txt`.

This capstone is a local, deterministic support assistant for policy questions and support-ticket lookups. It combines a seeded ticket dataset, a twelve-document policy knowledge base, two retrieval indexes, a three-agent CrewAI workflow, session memory, guardrails, an AutoGen review stage, evaluation, governance checks, and a FastAPI service.

The graded language-model paths use deterministic `MOCK_LLM` implementations and do not require a model API key. SentenceTransformers embeddings run locally. The embedding model must be present in the local Hugging Face cache before running in offline mode; no embedding weights are committed to this repository.

## Quick start

Python 3.10 or newer is required. From the repository root in PowerShell,
create an environment with an installed Python 3 version and install the pinned
dependencies:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
```

The RAG code runs in Hugging Face offline mode. If `all-MiniLM-L6-v2` is not already cached, download it once while online before using the project offline:

```powershell
.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Generate the dataset and run the reproducible checks and demos:

```powershell
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
```

Run the indexing, calibration, evaluation, and demo commands sequentially; they
share the local ChromaDB index and should not be run concurrently.

Run the API locally:

```powershell
.venv\Scripts\python.exe -m uvicorn ola_support_agent.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the HTTP API documentation. The service also exposes `WS /ws/chat` for multi-turn chat. CrewAI telemetry is disabled before imports with `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`.
For the customer-facing help-center preview, open `http://127.0.0.1:8000/`.
The home page lets a visitor ask a policy question, select a suggested topic,
optionally include a ticket ID, inspect source documents, and start a fresh
conversation. `/docs` remains available for API development and testing.

## Requirements and evidence map

| Capstone area | Implementation | Demonstration / evidence |
| --- | --- | --- |
| Task 1 — seeded ticket dataset and validation | `src/ola_support_agent/dataset.py`, `scripts/generate_dataset.py` | `transcripts/dataset_validation.txt`, `transcripts/dataset.json` |
| Task 2 — twelve policy documents and coverage check | `kb/`, `scripts/check_kb.py` | `transcripts/kb_check.txt` |
| Task 3 — fixed-size and sentence chunking, separate ChromaDB collections | `src/ola_support_agent/rag_core.py` | `transcripts/chunk_eval.txt` |
| Task 4 — calibrated grounded generation and fallback | `src/ola_support_agent/grounded_generation.py`, `src/ola_support_agent/guardrails.py`, `scripts/calibrate.py` | `config/threshold.json`, `transcripts/calibration.txt`, `transcripts/grounded_demo.txt`, `transcripts/guardrails.txt` |
| Task 5 — per-query precision/recall comparison | `src/ola_support_agent/evaluation.py` | `transcripts/chunk_eval.txt` |
| Task 6 — ticket lookup and designed escalation score | `src/ola_support_agent/tools.py` | `transcripts/crew_demo.txt` |
| Task 7 — three-agent CrewAI tool workflow | `src/ola_support_agent/crew_runner.py`, `src/ola_support_agent/tools.py` | `scripts/crew_demo.py`, `transcripts/crew_demo.txt` |
| Task 8 — multi-turn and fresh-session memory | `src/ola_support_agent/memory.py` | `transcripts/memory_demo.txt`, `transcripts/memory_fresh_session.txt` |
| Task 9 — structured CrewAI response schema | `src/ola_support_agent/schemas.py` | `transcripts/crew_demo.txt` (schema validation) |
| Task 10 — PII, injection, and groundedness guardrails | `src/ola_support_agent/guardrails.py` | `transcripts/guardrails.txt` |
| Tasks 11–12 — FastAPI, JSONL request logging, and WebSocket | `src/ola_support_agent/api.py` | `tests/`, API route docs at `/docs` |
| Task 13 — four-score, 15-query deterministic `MOCK_LLM` judge prompt | `src/ola_support_agent/evaluation_judge.py` | `transcripts/evaluation_judge.txt`, `transcripts/judge_eval.json` |
| Task 14 — AutoGen review and structured verdicts | `src/ola_support_agent/autogen_review.py` | `transcripts/autogen_review.txt` |
| Task 15 — least autonomy, risk classification, and request budget | `src/ola_support_agent/governance.py` | `transcripts/governance.txt` |
| Task 16 — repeated-query response cache | `src/ola_support_agent/grounded_generation.py` | `transcripts/grounded_demo.txt` (cache miss then hit) |

Regenerate the transcripts by rerunning the corresponding commands above. The `chroma_db/` index is generated locally and is intentionally excluded from Git; it is rebuilt from `kb/` when the app starts.

## Ticket lookup and escalation

The ticket escalation score is `0.6 × escalated + 0.4 × ((30 − days_since_created) / 30)`, clamped to `[0, 1]`. The recommended escalation threshold is **0.65**. With this formula, a ticket must be escalated and no more than 26 days old to meet the threshold.

## Retrieval and evaluation

The twelve `kb/` documents cover all required policy topics. `RAGCore` indexes both 500-character chunks with 100-character overlap and sentence-based chunks in separate cosine-similarity ChromaDB collections, using the local `all-MiniLM-L6-v2` SentenceTransformers model.

Calibration measured five in-scope top-1 cosine scores (**0.4530, 0.6897, 0.7155, 0.6460, 0.5662**) and two out-of-scope scores (**0.1095, 0.0389**). The configured threshold is **0.2812**, the midpoint between the closest observed in-scope and out-of-scope scores. These are empirical results for the included calibration queries and model; they are not a guarantee for every query.

The document-level chunking comparison reports fixed-size **P=0.1339, R=1.0000, F1=0.2335** and sentence-based **P=0.2252, R=1.0000, F1=0.3633**. Sentence-based chunking is selected because it improves precision and F1 while retaining full recall on this evaluation set. Precision remains modest, so some retrieved documents are irrelevant; the scores and per-query arithmetic are retained in `transcripts/chunk_eval.txt`.

The 15-query judge set includes all twelve policy topics, an edge case, an out-of-scope query, and an injection attempt. Each item is sent through the explicit Accuracy, Grounding, Completeness, and Safety prompt and validated against a structured score schema. Under `MOCK_LLM`, `MockJudge` returns deterministic scores using token-overlap, citation-overlap, fallback, and unsafe-claim checks. These are transparent lexical mock scores, not semantic judgments from a live LLM; no hosted model API is used. See the per-query results and four averages in `transcripts/evaluation_judge.txt` and `transcripts/judge_eval.json`.

On the current 15-query run, the measured averages are **Accuracy 0.5060,
Grounding 1.0000, Completeness 0.6278, and Safety 1.0000**. In particular, the
perfect Grounding and Safety values are outcomes of the mock's lexical/canned
checks, not a claim of semantic correctness or a production safety guarantee.

## Agent workflow, safety, and governance

- The Retrieval Agent owns `rag_lookup`, the Lookup Agent owns `check_support_ticket_status`, and the Response Composer has no tools. `scripts/crew_demo.py` invokes `.kickoff()` for a policy query and a different ticket query.
- Each crew response is validated against the Pydantic `CrewResponse` schema.
- Session memory is process-local. The multi-turn transcript shows a follow-up using the prior customer message; the separate fresh-session transcript shows the same follow-up without prior state.
- Input guardrails mask fixed-format phone numbers and reject prompt-injection patterns. The default output-groundedness guardrail and `GroundedGenerator` both load the calibrated threshold from `config/threshold.json`; an explicit threshold override remains supported. The demo transcript records the configured threshold and a sample override. The masker is intentionally scoped to the fixed-format phone field; free-text names, addresses, and payment details are not reliably identifiable by this simple pattern matcher.
- The AutoGen review uses two agents in a `RoundRobinGroupChat`, bounded to two turns, with structured `VerdictModel` output. Its transcript shows one unchanged approval and one correction of a deliberately unsupported refund promise.
- Governance checks restrict each tool to its designated agent, classify this customer-support system as Medium risk, and reject a request above the 4,000-token estimated budget.
- The FastAPI layer provides `POST /ask`, `POST /add-document`, health and cache endpoints, and `WS /ws/chat`. Request logs are JSON Lines with trace IDs, durations, and masked query text. The in-memory response cache is scoped by session, normalized query, and ticket ID.

## Repository layout

```text
config/                     Calibrated retrieval threshold
kb/                         Twelve support-policy documents
scripts/                    Dataset, calibration, retrieval, and CrewAI demos
src/ola_support_agent/      API, agents, tools, RAG, safety, review, evaluation
tests/                      Focused unit and API behavior tests
transcripts/                Reproducible output evidence for the capstone tasks
```

All examples use fictional support data. The project makes no external language-model API calls in its graded `MOCK_LLM` configuration.
