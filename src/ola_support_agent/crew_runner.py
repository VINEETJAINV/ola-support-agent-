"""
Crew runner for executing CrewAI queries.
"""
import os
import json
from typing import Dict, Any, Optional, ClassVar, List

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ola_support_agent.mock_llm import create_mock_llm
from ola_support_agent.tools import check_support_ticket_status
from ola_support_agent.rag_core import RAGCore
from ola_support_agent.grounded_generation import GroundedGenerator
from ola_support_agent.guardrails import GuardrailPipeline
from ola_support_agent.schemas import CrewResponse


_CURRENT_QUERY_MARKER = "Current user query: "
_RETRIEVAL_CONTEXT_MARKER = "Prior user queries for retrieval context: "


class RAGLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search the Ola support knowledge base for policy information."
    
    class ArgsSchema(BaseModel):
        query: str = Field(..., description="The question or query to search the knowledge base for")
    
    args_schema: type = ArgsSchema
    last_citations: ClassVar[List[Dict[str, Any]]] = []
    
    def _run(self, query: str) -> str:
        rag = RAGCore()
        generator = GroundedGenerator(rag, collection_name="sentence")
        current_query = query
        retrieval_context = ""
        if _CURRENT_QUERY_MARKER in query:
            search_input = query.split(_CURRENT_QUERY_MARKER, 1)[1]
            if _RETRIEVAL_CONTEXT_MARKER in search_input:
                current_query, retrieval_context = search_input.split(
                    _RETRIEVAL_CONTEXT_MARKER, 1
                )
            else:
                current_query = search_input
            current_query = current_query.strip()
            retrieval_context = retrieval_context.split(
                "\nAnswer the current query", 1
            )[0].strip()
        response = generator.generate(current_query, context=retrieval_context)
        type(self).last_citations = response.citations
        if response.fallback_triggered:
            return "I don't have enough information in the knowledge base to answer this question."
        result = f"Answer: {response.answer}\n\nCitations:\n"
        for cite in response.citations:
            result += f"- [{cite['doc_id']}] (score: {cite['score']:.3f}): {cite['text_snippet'][:150]}...\n"
        return result


class CheckTicketStatusTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = "Look up a specific support ticket by its record ID."
    
    class ArgsSchema(BaseModel):
        record_id: str = Field(..., description="The ticket record ID to look up (format: TKT-XXXX)")
    
    args_schema: type = ArgsSchema
    last_result: ClassVar[Dict[str, Any]] = {}
    
    def _run(self, record_id: str) -> str:
        result = check_support_ticket_status(record_id)
        type(self).last_result = result
        if "error" in result:
            return f"Error: {result['error']}"
        return (
            f"Ticket: {result['record_id']}\n"
            f"Status: {result['status']}\n"
            f"Resolution Time: {result['resolution_time_hours']} hours\n"
            f"Category: {result['category']}\n"
            f"Days Since Created: {result['days_since_created']}\n"
            f"Escalated: {result['escalated']}\n"
            f"Escalation Score: {result['escalation_score']}\n"
            f"Escalation Recommended: {result['escalation_recommended']}"
        )


async def run_crew_query(
    user_query: str,
    ticket_id: Optional[str] = None,
    context: str = "",
    retrieval_context: str = "",
) -> Dict[str, Any]:
    """Run a crew query with the mock LLM."""
    CheckTicketStatusTool.last_result = {}
    RAGLookupTool.last_citations = []
    
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Search the Ola support knowledge base for policy information",
        backstory="You are a specialized retrieval agent with access to Ola's support policy knowledge base.",
        tools=[RAGLookupTool()],
        llm=create_mock_llm(),
        verbose=False,
    )
    
    lookup_agent = Agent(
        role="Ticket Lookup Agent",
        goal="Look up specific support ticket statuses and compute escalation scores",
        backstory="You can access the live support ticket database and compute escalation scores.",
        tools=[CheckTicketStatusTool()],
        llm=create_mock_llm(),
        verbose=False,
    )
    
    composer_agent = Agent(
        role="Response Composer",
        goal="Combine information from retrieval and lookup agents into a clear final response",
        backstory="You craft complete, accurate, and empathetic responses for customers.",
        tools=[],
        llm=create_mock_llm(),
        verbose=False,
    )
    
    tasks = []
    
    # Retrieval task
    context_section = (
        f"Prior conversation context (use only to resolve references):\n{context}\n\n"
        if context else ""
    )
    retrieval_task = Task(
        description=(
            f"{context_section}Search the knowledge base using this input:\n"
            f"{_CURRENT_QUERY_MARKER}{user_query}\n"
            f"{_RETRIEVAL_CONTEXT_MARKER}{retrieval_context}\n"
            "Answer the current query, keeping "
            "its wording and intent distinct from prior context. Return relevant policy "
            "details with citations."
        ),
        expected_output="Policy information with document citations relevant to the query",
        agent=retrieval_agent
    )
    tasks.append(retrieval_task)
    
    # Lookup task (if ticket_id provided)
    if ticket_id:
        lookup_task = Task(
            description=f"Look up ticket {ticket_id} and return its status, resolution time, and escalation score with recommendation.",
            expected_output="Ticket status, resolution time, escalation score, and recommendation",
            agent=lookup_agent
        )
        tasks.append(lookup_task)
    
    # Compose task
    compose_task = Task(
        description=(
            f"{context_section}Compose a final response to the current customer query: "
            f"'{user_query}'. Use prior context only to interpret follow-up references; "
            "do not replace or broaden the current query. Use policy information from "
            "retrieval and ticket data from lookup (if available). Provide a clear, "
            "helpful answer with citations where appropriate."
        ),
        expected_output="Complete customer-facing response with policy citations and ticket data if applicable",
        agent=composer_agent
    )
    tasks.append(compose_task)
    
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent] if ticket_id else [retrieval_agent, composer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )
    
    result = await crew.kickoff_async()
    
    # Extract ticket info if available
    ticket_info = None
    if ticket_id:
        ticket_result = CheckTicketStatusTool.last_result
        if "error" not in ticket_result:
            ticket_info = ticket_result
    
    citations = RAGLookupTool.last_citations
    response = CrewResponse(
        answer=str(result),
        citations=citations,
        ticket_info=ticket_info,
        confidence=citations[0]["score"] if citations else 0.0,
        fallback_triggered=not bool(citations),
        session_id="crew-demo",
        turn_number=1,
    )
    return {**response.model_dump(), "response_format": response.model_dump()}
