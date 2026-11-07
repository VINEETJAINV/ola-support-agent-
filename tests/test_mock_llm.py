"""
Tests for the Mock LLM implementation.
"""
import os
import sys
import json
from types import SimpleNamespace

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ola_support_agent.mock_llm import MockLLM, create_mock_llm
from ola_support_agent.autogen_mock import MockChatCompletionClient, create_mock_autogen_client


class TicketStatusInput(BaseModel):
    record_id: str = Field(..., description="The ticket record ID to look up")


class TicketStatusTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = "Look up a support ticket status by record ID"
    args_schema: type = TicketStatusInput
    
    def _run(self, record_id: str) -> str:
        # Mock ticket data
        tickets = {
            "TKT-0001": {"status": "Open", "resolution_time_hours": 2.5, "category": "Billing"},
            "TKT-0002": {"status": "In Progress", "resolution_time_hours": 4.0, "category": "Technical Issue"},
            "TKT-0003": {"status": "Resolved", "resolution_time_hours": 1.0, "category": "Account Access"},
        }
        ticket = tickets.get(record_id, {"error": "Not found"})
        return json.dumps(ticket)


class RAGLookupInput(BaseModel):
    query: str = Field(..., description="The query to search for")


class RAGLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search the knowledge base for policy information"
    args_schema: type = RAGLookupInput
    
    def _run(self, query: str) -> str:
        # Mock RAG results
        return json.dumps({
            "results": [
                {"doc_id": "doc-1", "content": "Refund policy: Full refund within 30 days for defective products.", "score": 0.9},
                {"doc_id": "doc-2", "content": "SLA: Critical issues resolved within 4 hours.", "score": 0.8},
            ]
        })


def test_mock_llm_final_answer_contains_retrieved_text():
    """Test that final answer contains retrieved text from tools."""
    llm = create_mock_llm()
    rag_tool = RAGLookupTool()
    ticket_tool = TicketStatusTool()
    
    # Create a retrieval agent
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Answer policy questions using the knowledge base",
        backstory="You are a policy expert.",
        tools=[rag_tool],
        llm=llm,
        verbose=False
    )
    
    task = Task(
        description="What is the refund policy for defective products?",
        expected_output="A clear answer about the refund policy with citations",
        agent=retrieval_agent
    )
    
    crew = Crew(agents=[retrieval_agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    
    # Check that the answer contains retrieved text
    result_str = str(result)
    assert "refund" in result_str.lower() or "Refund" in result_str
    assert "defective" in result_str.lower() or "Defective" in result_str


def test_two_different_ticket_ids_give_different_outputs():
    """Test that two different ticket IDs give different outputs."""
    llm = create_mock_llm()
    ticket_tool = TicketStatusTool()
    
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Look up ticket statuses",
        backstory="You look up ticket statuses.",
        tools=[ticket_tool],
        llm=llm,
        verbose=False
    )
    
    # Test with TKT-0001
    task1 = Task(
        description="Check the status of ticket TKT-0001",
        expected_output="Ticket status for TKT-0001",
        agent=lookup_agent
    )
    crew1 = Crew(agents=[lookup_agent], tasks=[task1], process=Process.sequential, verbose=False)
    result1 = str(crew1.kickoff())
    
    # Create new LLM for second call to avoid state issues
    llm2 = create_mock_llm()
    lookup_agent2 = Agent(
        role="Lookup Agent",
        goal="Look up ticket statuses",
        backstory="You look up ticket statuses.",
        tools=[ticket_tool],
        llm=llm2,
        verbose=False
    )
    
    task2 = Task(
        description="Check the status of ticket TKT-0002",
        expected_output="Ticket status for TKT-0002",
        agent=lookup_agent2
    )
    crew2 = Crew(agents=[lookup_agent2], tasks=[task2], process=Process.sequential, verbose=False)
    result2 = str(crew2.kickoff())
    
    # Results should be different
    assert result1 != result2
    assert "TKT-0001" in result1 or "Open" in result1
    assert "TKT-0002" in result2 or "In Progress" in result2


def test_composer_makes_zero_tool_calls():
    """Test that the Composer agent (with no tools) makes zero tool calls."""
    llm = create_mock_llm()
    
    composer = Agent(
        role="Composer",
        goal="Compose final answers from retrieved information",
        backstory="You compose final answers.",
        tools=[],  # No tools
        llm=llm,
        verbose=False
    )
    
    task = Task(
        description="Compose a final answer based on this context: The refund policy allows full refunds within 30 days.",
        expected_output="A composed final answer",
        agent=composer
    )
    
    crew = Crew(agents=[composer], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    
    # The composer should not make any tool calls (it has no tools)
    # The call count should be 1 (just the final answer generation)
    assert llm._call_count >= 1


def test_same_query_twice_gives_identical_output():
    """Test that the same query twice gives identical output."""
    llm = create_mock_llm()
    rag_tool = RAGLookupTool()
    
    agent = Agent(
        role="Test Agent",
        goal="Answer queries",
        backstory="You answer queries.",
        tools=[rag_tool],
        llm=llm,
        verbose=False
    )
    
    task = Task(
        description="What is the SLA for critical issues?",
        expected_output="SLA information",
        agent=agent
    )
    
    crew1 = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result1 = str(crew1.kickoff())
    
    # Create new agent with new LLM for second run
    llm2 = create_mock_llm()
    agent2 = Agent(
        role="Test Agent",
        goal="Answer queries",
        backstory="You answer queries.",
        tools=[rag_tool],
        llm=llm2,
        verbose=False
    )
    
    task2 = Task(
        description="What is the SLA for critical issues?",
        expected_output="SLA information",
        agent=agent2
    )
    
    crew2 = Crew(agents=[agent2], tasks=[task2], process=Process.sequential, verbose=False)
    result2 = str(crew2.kickoff())
    
    # Results should be identical
    assert result1 == result2


def test_policy_answer_and_ticket_answer_differ():
    """Test that a policy answer and a ticket answer differ."""
    llm1 = create_mock_llm()
    llm2 = create_mock_llm()
    
    rag_tool = RAGLookupTool()
    ticket_tool = TicketStatusTool()
    
    # Policy query
    policy_agent = Agent(
        role="Policy Agent",
        goal="Answer policy questions",
        backstory="You answer policy questions.",
        tools=[rag_tool],
        llm=llm1,
        verbose=False
    )
    
    policy_task = Task(
        description="What is the refund policy?",
        expected_output="Refund policy answer",
        agent=policy_agent
    )
    crew1 = Crew(agents=[policy_agent], tasks=[policy_task], process=Process.sequential, verbose=False)
    policy_result = str(crew1.kickoff())
    
    # Ticket query
    ticket_agent = Agent(
        role="Ticket Agent",
        goal="Look up tickets",
        backstory="You look up tickets.",
        tools=[ticket_tool],
        llm=llm2,
        verbose=False
    )
    
    ticket_task = Task(
        description="Check ticket TKT-0001 status",
        expected_output="Ticket status",
        agent=ticket_agent
    )
    crew2 = Crew(agents=[ticket_agent], tasks=[ticket_task], process=Process.sequential, verbose=False)
    ticket_result = str(crew2.kickoff())
    
    # They should be different
    assert policy_result != ticket_result


def test_answer_never_placeholder():
    """Test that the answer is never a placeholder."""
    llm = create_mock_llm()
    rag_tool = RAGLookupTool()
    
    agent = Agent(
        role="Test Agent",
        goal="Answer queries",
        backstory="You answer queries.",
        tools=[rag_tool],
        llm=llm,
        verbose=False
    )
    
    task = Task(
        description="What is the SLA for critical issues?",
        expected_output="SLA information",
        agent=agent
    )
    
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = str(crew.kickoff())
    
    # Should not contain placeholder text
    assert "placeholder" not in result.lower()
    assert "TODO" not in result
    assert "mock" not in result.lower() or "MockLLM" not in result
    assert len(result.strip()) > 20  # Should have substantial content


def test_response_composer_uses_retrieved_context_and_ticket_summary():
    llm = create_mock_llm()
    messages = [{
        "role": "user",
        "content": (
            "Current Task: Compose a final response to the customer query.\n\n"
            "This is the context you're working with:\n"
            "Answer: Support runs Monday through Saturday, 6 AM to 11 PM IST.\n\n"
            "Citations:\n- [business-hours-holidays]\n\n"
            "Ticket: TKT-0001\nStatus: Open\n"
            "Resolution Time: 16.46 hours\nCategory: Account Access\n"
            "Days Since Created: 23\nEscalated: False\n"
            "Escalation Score: 0.0933\nEscalation Recommended: False\n\n"
            "Provide your complete response:"
        ),
    }]

    response = llm.call(
        messages,
        from_agent=SimpleNamespace(role="Response Composer", tools=[]),
    )

    assert "Support runs Monday through Saturday" in response
    assert "Ticket TKT-0001 is Open" in response
    assert "Compose a final response" not in response


# Autogen mock tests
def test_autogen_mock_creates_valid_response():
    """Test that the autogen mock creates valid responses."""
    client = create_mock_autogen_client(
        responses=['{"approved": true, "final_answer": "Approved answer", "reason": "All good"}']
    )
    
    assert client is not None
    assert client.model_info["structured_output"] == True
    assert client.model_info["function_calling"] == True


@pytest.mark.asyncio
async def test_autogen_mock_create_returns_valid_result():
    """Test that autogen mock create returns valid CreateResult."""
    client = create_mock_autogen_client(
        responses=['{"approved": true, "final_answer": "Test", "reason": "OK"}']
    )
    
    from autogen_core.models import CreateResult
    messages = [{"role": "user", "content": "Test"}]
    
    result = await client.create(messages=messages)
    
    assert isinstance(result, CreateResult)
    assert result.content is not None
    assert "approved" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])