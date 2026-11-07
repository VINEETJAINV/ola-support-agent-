"""
FastAPI application for Ola Support Agent.
"""
import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ola_support_agent.rag_core import RAGCore
from ola_support_agent.grounded_generation import GroundedGenerator
from ola_support_agent.guardrails import GuardrailPipeline, PIIMasker
from ola_support_agent.memory import SessionMemory
from ola_support_agent.crew_runner import run_crew_query
from ola_support_agent.autogen_review import run_review_stage


# Pydantic models
class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    ticket_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    ticket_info: Optional[Dict[str, Any]] = None
    confidence: float
    fallback_triggered: bool
    session_id: str
    turn_number: int
    trace_id: str


class AddDocumentRequest(BaseModel):
    title: str
    content: str


class AddDocumentResponse(BaseModel):
    doc_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# Global state
rag_core: RAGCore = None
grounded_generator: GroundedGenerator = None
guardrail_pipeline: GuardrailPipeline = None
session_memory: SessionMemory = None
response_cache: Dict[str, Any] = {}
cache_stats = {"hits": 0, "misses": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_core, grounded_generator, guardrail_pipeline, session_memory
    
    rag_core = RAGCore()
    rag_core.index_documents()
    grounded_generator = GroundedGenerator(rag_core)
    guardrail_pipeline = GuardrailPipeline(groundedness_threshold=grounded_generator.threshold)
    session_memory = SessionMemory()
    
    print("FastAPI application started")
    yield
    
    print("FastAPI application shutting down")


app = FastAPI(
    title="Ola Support Agent API",
    description="Domain support agent for Ola customer support",
    version="1.0.0",
    lifespan=lifespan
)


# Setup JSONL logging
logger = logging.getLogger("ola_support")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def log_request(
    trace_id: str,
    method: str,
    path: str,
    request_data: Dict,
    response_data: Dict,
    duration_ms: float,
    masked_request: str = None
):
    log_entry = {
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": method,
        "path": path,
        "duration_ms": round(duration_ms, 2),
        "request": masked_request or request_data,
        "response_status": response_data.get("status", "unknown"),
        "response_size": len(str(response_data))
    }
    logger.info(json.dumps(log_entry))


def get_trace_id() -> str:
    return str(uuid.uuid4())[:8]


def mask_for_logging(text: str) -> str:
    """Mask phone numbers for logging."""
    return PIIMasker.mask(text)[0]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    started = time.time()
    response = HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.utcnow().isoformat())
    log_request(get_trace_id(), "GET", "/health", {}, response.model_dump(), (time.time() - started) * 1000)
    return response


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, http_request: Request):
    start_time = time.time()
    trace_id = get_trace_id()
    
    session_id = request.session_id or f"session-{trace_id}"
    user_query = request.query
    ticket_id = request.ticket_id
    
    # Input guardrails
    masked_query, guardrail_result = guardrail_pipeline.process_input(user_query)
    
    if not guardrail_result.passed:
        response = AskResponse(
            answer="I cannot process this request due to security policy violation.",
            confidence=0.0,
            fallback_triggered=True,
            session_id=session_id,
            turn_number=1,
            trace_id=trace_id
        )
        log_request(trace_id, "POST", "/ask", {"query": mask_for_logging(user_query)}, 
                   response.model_dump(), (time.time() - start_time) * 1000, mask_for_logging(user_query))
        return response
    
    # Budget check (approximate token count)
    estimated_tokens = len(user_query) // 4 + 1000  # rough estimate
    if estimated_tokens > 4000:
        response = AskResponse(
            answer="Request exceeds token budget limit.",
            confidence=0.0,
            fallback_triggered=True,
            session_id=session_id,
            turn_number=1,
            trace_id=trace_id
        )
        log_request(trace_id, "POST", "/ask", {"query": mask_for_logging(user_query)}, 
                   response.model_dump(), (time.time() - start_time) * 1000, mask_for_logging(user_query))
        return response
    
    # Check cache
    cache_key = f"{session_id}:{' '.join(masked_query.casefold().split())}:{ticket_id or ''}"
    if cache_key in response_cache:
        cache_stats["hits"] += 1
        cached = response_cache[cache_key]
        cached["trace_id"] = trace_id
        response = AskResponse(**cached)
        log_request(trace_id, "POST", "/ask", {"query": mask_for_logging(user_query)}, 
                   response.model_dump(), (time.time() - start_time) * 1000, mask_for_logging(user_query))
        return response
    
    cache_stats["misses"] += 1
    
    # Get conversation history
    history = session_memory.get_history(session_id)
    turn_number = len(history) // 2 + 1
    
    # Run crew query
    crew_result = await run_crew_query(masked_query, ticket_id)
    
    # Run autogen review
    # Get retrieved context for review
    retrieved = rag_core.retrieve(masked_query, collection_name="sentence", top_k=5)
    retrieved_context = "\n".join([r["text"] for r in retrieved])
    
    review_result = await run_review_stage(crew_result["answer"], retrieved_context)
    final_answer = review_result.final_answer
    review_approved = review_result.approved
    
    # Get ticket info if requested
    ticket_info = None
    if ticket_id:
        ticket_info = crew_result.get("ticket_info")
    
    # Get citations from grounded generation
    gen_result = grounded_generator.generate(masked_query)
    citations = gen_result.citations
    
    # Groundedness check
    grounded_check = guardrail_pipeline.process_output(masked_query, final_answer, 
        retrieved)
    
    if not grounded_check.passed:
        final_answer = "I don't have enough information to answer this question based on the available knowledge base."
    
    response = AskResponse(
        answer=final_answer,
        citations=gen_result.citations,
        ticket_info=ticket_info,
        confidence=gen_result.confidence,
        fallback_triggered=gen_result.fallback_triggered,
        session_id=session_id,
        turn_number=turn_number,
        trace_id=trace_id
    )
    
    # Cache response
    response_cache[cache_key] = response.model_dump()
    
    # Update session memory
    session_memory.add_message(session_id, "human", masked_query)
    session_memory.add_message(session_id, "ai", final_answer)
    
    log_request(trace_id, "POST", "/ask", {"query": mask_for_logging(user_query)}, 
               response.model_dump(), (time.time() - start_time) * 1000, mask_for_logging(user_query))
    
    return response


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest, http_request: Request):
    start_time = time.time()
    trace_id = get_trace_id()
    
    doc_id = request.title.lower().replace(" ", "-").replace("_", "-")
    file_path = f"./kb/{doc_id}.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(request.content)
    
    # Re-index
    rag = RAGCore()
    rag.index_documents()
    
    response = AddDocumentResponse(doc_id=doc_id, message="Document added and indexed successfully")
    
    log_request(trace_id, "POST", "/add-document", {"title": mask_for_logging(request.title)}, 
               response.model_dump(), (time.time() - start_time) * 1000)
    
    return response


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    trace_id = get_trace_id()
    
    try:
        while True:
            trace_id = get_trace_id()
            data = await websocket.receive_text()
            start_time = time.time()
            
            try:
                request_data = json.loads(data)
                user_query = request_data.get("query", "")
                ticket_id = request_data.get("ticket_id")
            except json.JSONDecodeError:
                user_query = data
                ticket_id = None

            estimated_tokens = len(user_query) // 4 + 1000
            if estimated_tokens > 4000:
                response = {"error": "Request exceeds the 4,000-token budget.", "status_code": 413, "trace_id": trace_id}
                await websocket.send_text(json.dumps(response))
                log_request(trace_id, "WS", "/ws/chat", {"query": mask_for_logging(user_query)}, response,
                            (time.time() - start_time) * 1000, mask_for_logging(user_query))
                continue
            
            # Input guardrails
            masked_query, guardrail_result = guardrail_pipeline.process_input(user_query)
            
            if not guardrail_result.passed:
                response = {
                    "answer": "I cannot process this request due to security policy violation.",
                    "confidence": 0.0,
                    "fallback_triggered": True,
                    "session_id": session_id,
                    "trace_id": trace_id
                }
                await websocket.send_text(json.dumps(response))
                log_request(trace_id, "WS", "/ws/chat", {"query": mask_for_logging(user_query)}, response,
                            (time.time() - start_time) * 1000, mask_for_logging(user_query))
                continue
            
            # Check cache
            cache_key = f"{session_id}:{' '.join(masked_query.casefold().split())}:{ticket_id or ''}"
            if cache_key in response_cache:
                cache_stats["hits"] += 1
                cached = response_cache[cache_key]
                cached["trace_id"] = trace_id
                await websocket.send_text(json.dumps(cached))
                log_request(trace_id, "WS", "/ws/chat", {"query": mask_for_logging(user_query)}, cached,
                            (time.time() - start_time) * 1000, mask_for_logging(user_query))
                continue
            
            cache_stats["misses"] += 1
            
            # Run crew query
            crew_result = await run_crew_query(masked_query, ticket_id)
            
            # Run autogen review
            retrieved = rag_core.retrieve(masked_query, collection_name="sentence", top_k=5)
            retrieved_context = "\n".join([r["text"] for r in retrieved])
            
            review_result = await run_review_stage(crew_result["answer"], retrieved_context)
            final_answer = review_result.final_answer
            
            # Get ticket info
            ticket_info = None
            if ticket_id:
                ticket_info = crew_result.get("ticket_info")
            
            # Get citations
            gen_result = grounded_generator.generate(masked_query)
            citations = gen_result.citations
            
            # Groundedness check
            grounded_check = guardrail_pipeline.process_output(masked_query, final_answer, 
                retrieved)
            
            if not grounded_check.passed:
                final_answer = "I don't have enough information to answer this question based on the available knowledge base."
            
            response = {
                "answer": final_answer,
                "citations": gen_result.citations,
                "ticket_info": ticket_info,
                "confidence": gen_result.confidence,
                "fallback_triggered": gen_result.fallback_triggered,
                "session_id": session_id,
                "trace_id": trace_id
            }
            
            response_cache[cache_key] = response
            session_memory.add_message(session_id, "human", masked_query)
            session_memory.add_message(session_id, "ai", final_answer)
            
            await websocket.send_text(json.dumps(response))
            
            log_request(trace_id, "WS", "/ws/chat", {"query": mask_for_logging(user_query)}, 
                       response, (time.time() - start_time) * 1000, mask_for_logging(user_query))
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass


@app.get("/cache/stats")
async def cache_stats_endpoint():
    started = time.time()
    result = {
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "hit_rate": cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0,
        "size": len(response_cache)
    }
    log_request(get_trace_id(), "GET", "/cache/stats", {}, result, (time.time() - started) * 1000)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
