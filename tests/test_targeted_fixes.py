import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ola_support_agent import api
from ola_support_agent.dataset import AMOUNT_MAX_INR, AMOUNT_MIN_INR, generate_tickets
from ola_support_agent.evaluation_judge import (
    JudgeScores,
    MockJudge,
    build_judge_prompt,
)
from ola_support_agent.grounded_generation import GroundedGenerator
from ola_support_agent.guardrails import GuardrailPipeline, load_threshold
from ola_support_agent.memory import SessionMemory
from ola_support_agent.rag_core import FixedSizeChunker, RAGCore, SentenceChunker


def test_grounded_generator_filters_unrelated_retrieved_chunks():
    class FakeRAG:
        def retrieve(self, query, collection_name, top_k):
            assert query == "refund billing"
            return [
                {"doc_id": "unrelated", "chunk_id": "u1", "text": "SLA support response targets.",
                 "score": 0.91},
                {"doc_id": "billing", "chunk_id": "b1", "text": "Billing errors receive a refund.",
                 "score": 0.42},
            ]

    response = GroundedGenerator(FakeRAG(), threshold=0.3).generate("refund billing")

    assert not response.fallback_triggered
    assert "Billing errors receive a refund" in response.answer
    assert "SLA support response targets" not in response.answer
    assert [citation["doc_id"] for citation in response.citations] == ["billing"]


def test_grounded_generator_falls_back_without_query_overlap():
    class FakeRAG:
        def retrieve(self, query, collection_name, top_k):
            return [{"doc_id": "other", "chunk_id": "o1", "text": "SLA response windows.",
                     "score": 0.95}]

    response = GroundedGenerator(FakeRAG(), threshold=0.3).generate("refund billing")
    assert response.fallback_triggered
    assert response.citations == []


def test_current_query_match_takes_priority_over_historical_context():
    class FakeRAG:
        def retrieve(self, query, collection_name, top_k):
            return [
                {"doc_id": "current", "chunk_id": "c1", "text": "Response times are under 2 minutes.",
                 "score": 0.6},
                {"doc_id": "history", "chunk_id": "h1", "text": "Support retention lasts seven years.",
                 "score": 0.9},
            ]

    response = GroundedGenerator(FakeRAG(), threshold=0.3).generate(
        "How fast is the response?", context="Which support policy applies?"
    )
    assert [citation["doc_id"] for citation in response.citations] == ["current"]


class _FakeCollection:
    def __init__(self):
        self.rows = {}
        self.events = []

    def delete(self, where):
        self.events.append(("delete", where["doc_id"]))
        self.rows = {
            chunk_id: row for chunk_id, row in self.rows.items()
            if row["metadata"]["doc_id"] != where["doc_id"]
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        self.events.append(("upsert", list(ids)))
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[chunk_id] = {"document": document, "metadata": metadata}


class _FakeEmbeddings(list):
    def tolist(self):
        return list(self)


def test_reindex_removes_obsolete_chunks_for_changed_document():
    rag = RAGCore.__new__(RAGCore)
    rag.fixed_collection = _FakeCollection()
    rag.sentence_collection = _FakeCollection()
    rag.fixed_chunker = FixedSizeChunker(chunk_size=5, overlap=0)
    rag.sentence_chunker = SentenceChunker()
    rag.model = SimpleNamespace(encode=lambda texts: _FakeEmbeddings([[1] for _ in texts]))
    documents = {"policy": "First sentence. Second obsolete sentence."}
    rag.load_knowledge_base = lambda _kb_dir: documents.copy()

    rag.index_documents()
    assert any("obsolete" in row["document"] for row in rag.sentence_collection.rows.values())

    documents["policy"] = "Current."
    stats = rag.index_documents()

    assert stats == {"fixed_chunks": 2, "sentence_chunks": 1}
    assert all("obsolete" not in row["document"] for row in rag.sentence_collection.rows.values())
    assert len(rag.sentence_collection.rows) == 1
    assert len(rag.fixed_collection.rows) == 2
    assert rag.sentence_collection.events[-2][0] == "delete"


def test_add_document_rejects_unsafe_titles_and_writes_safe_title(tmp_path, monkeypatch):
    class FakeRAG:
        def index_documents(self):
            return {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "RAGCore", FakeRAG)
    (tmp_path / "kb").mkdir()

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.add_document(api.AddDocumentRequest(title="../escape", content="x"), None))
    assert error.value.status_code == 400
    assert not (tmp_path / "escape.txt").exists()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("unchanged", encoding="utf-8")
    try:
        (tmp_path / "kb" / "linked-policy.txt").symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform.")
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.add_document(
            api.AddDocumentRequest(title="linked-policy", content="overwrite"), None
        ))
    assert error.value.status_code == 400
    assert outside_file.read_text(encoding="utf-8") == "unchanged"

    result = asyncio.run(api.add_document(
        api.AddDocumentRequest(title="New Policy", content="Safe text"), None
    ))
    assert result.doc_id == "new-policy"
    assert (tmp_path / "kb" / "new-policy.txt").read_text(encoding="utf-8") == "Safe text"


def test_judge_uses_reference_overlap_citations_and_safety_rules():
    judge = MockJudge()
    reference = "Billing error excess amount refunded to original payment method"
    good = judge.score(
        "How are billing errors handled?",
        "Billing error excess amount refunded to original payment method.",
        [{"text_snippet": "Billing error excess amount refunded to original payment method"}],
        False,
        reference,
    )
    unsafe = judge.score(
        "How are refunds handled?", "We guarantee a refund.", [], False, reference
    )

    assert good == {"accuracy": 1.0, "grounding": 1.0, "completeness": 1.0, "safety": 1.0}
    assert unsafe["safety"] == 0.0
    assert unsafe["accuracy"] < good["accuracy"]
    assert unsafe["grounding"] == 0.0


def test_mock_llm_judge_returns_structured_scores_from_prompt():
    judge = MockJudge()
    payload = {
        "query": "How are billing errors handled?",
        "reference": "Billing error excess amount refunded",
        "answer": "Billing error excess amount refunded",
        "citations": [{"text_snippet": "Billing error excess amount refunded"}],
        "fallback": False,
    }
    prompt = build_judge_prompt(payload)

    scores = JudgeScores.model_validate_json(judge.call(prompt))

    assert set(scores.model_dump()) == {"accuracy", "grounding", "completeness", "safety"}
    assert scores.model_dump() == {
        "accuracy": 1.0,
        "grounding": 1.0,
        "completeness": 1.0,
        "safety": 1.0,
    }
    assert all(
        name in prompt
        for name in ("Accuracy", "Grounding", "Completeness", "Safety")
    )


def test_default_guardrail_uses_calibrated_config_outside_repo_cwd(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "config" / "threshold.json"
    expected = json.loads(config_path.read_text(encoding="utf-8"))["threshold"]
    monkeypatch.chdir(tmp_path)

    default_pipeline = GuardrailPipeline()
    override_pipeline = GuardrailPipeline(groundedness_threshold=0.9)
    default_generator = GroundedGenerator(SimpleNamespace())

    assert load_threshold() == expected
    assert default_pipeline.groundedness_checker.threshold == expected
    assert default_generator.threshold == expected
    assert override_pipeline.groundedness_checker.threshold == 0.9


def test_synthetic_ticket_amounts_are_seeded_and_within_documented_range():
    first = generate_tickets()
    second = generate_tickets()

    assert [ticket.amount_inr for ticket in first] == [
        ticket.amount_inr for ticket in second
    ]
    assert all(AMOUNT_MIN_INR <= ticket.amount_inr <= AMOUNT_MAX_INR for ticket in first)


def test_memory_transcripts_distinguish_followup_from_fresh_session():
    from ola_support_agent.memory import demonstrate_memory

    multi_turn, fresh_turn = demonstrate_memory()

    assert '"mock_llm_prior_user_messages": ["My refund is pending"]' in multi_turn
    assert "History before turn: []" in fresh_turn
    assert '"mock_llm_prior_user_messages": []' in fresh_turn


def test_support_home_serves_customer_facing_page_and_assets():
    response = asyncio.run(api.support_home())
    page_path = Path(response.path)
    html = page_path.read_text(encoding="utf-8")
    asset_routes = {route.path for route in api.app.routes}

    assert page_path.name == "index.html"
    assert "Ola Support Assistant" in html
    assert 'id="question"' in html
    assert "/assets/support.css" in html
    assert "/assets/support.js" in html
    assert "/assets" in asset_routes
    assert (page_path.parent / "support.css").is_file()
    assert (page_path.parent / "support.js").is_file()


def _install_api_fakes(monkeypatch):
    api.response_cache.clear()
    api.cache_stats.update(hits=0, misses=0)
    api.session_memory = SessionMemory()
    api.guardrail_pipeline = SimpleNamespace(
        process_input=lambda query: (query, SimpleNamespace(passed=True)),
        process_output=lambda query, answer, chunks: SimpleNamespace(passed=True),
    )
    api.rag_core = SimpleNamespace(retrieve=lambda query, **kwargs: [])
    api.run_review_stage = _fake_review
    api.grounded_generator = SimpleNamespace(
        generate=lambda query, context="": SimpleNamespace(
            citations=[], confidence=0.8, fallback_triggered=False
        )
    )
    captured = []

    async def crew_query(query, ticket_id=None, context="", retrieval_context=""):
        captured.append((query, context, retrieval_context))
        return {"answer": "Grounded answer", "ticket_info": None}

    monkeypatch.setattr(api, "run_crew_query", crew_query)
    return captured


async def _fake_review(answer, context):
    return SimpleNamespace(final_answer=answer, approved=True)


def test_api_passes_session_history_to_followup_and_isolates_sessions(monkeypatch):
    captured = _install_api_fakes(monkeypatch)

    asyncio.run(api.ask(api.AskRequest(query="My refund is pending", session_id="one"), None))
    asyncio.run(api.ask(api.AskRequest(query="How long will it take?", session_id="one"), None))
    asyncio.run(api.ask(api.AskRequest(query="How long will it take?", session_id="two"), None))

    assert captured[1][0] == "How long will it take?"
    assert "My refund is pending" in captured[1][1]
    assert captured[1][2] == "My refund is pending"
    assert captured[2] == ("How long will it take?", "", "")


def test_api_cache_hit_advances_turn_and_records_memory(monkeypatch):
    captured = _install_api_fakes(monkeypatch)

    first = asyncio.run(api.ask(api.AskRequest(query="What is the refund policy?", session_id="cache"), None))
    second = asyncio.run(api.ask(api.AskRequest(query="What is the refund policy?", session_id="cache"), None))

    assert first.turn_number == 1
    assert second.turn_number == 2
    assert len(api.session_memory.get_history("cache")) == 4
    assert len(captured) == 1
    assert api.cache_stats["hits"] == 1


def test_websocket_non_cached_responses_include_turn_number(monkeypatch):
    _install_api_fakes(monkeypatch)

    class FakeWebSocket:
        def __init__(self):
            self.messages = iter([
                json.dumps({"query": "First question"}),
                json.dumps({"query": "A distinct follow-up"}),
            ])
            self.responses = []

        async def accept(self):
            pass

        async def receive_text(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise WebSocketDisconnect(code=1000)

        async def send_text(self, message):
            self.responses.append(json.loads(message))

    websocket = FakeWebSocket()
    asyncio.run(api.websocket_chat(websocket))

    assert [response["turn_number"] for response in websocket.responses] == [1, 2]
