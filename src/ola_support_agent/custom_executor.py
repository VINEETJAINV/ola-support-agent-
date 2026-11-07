"""Backward-compatible executor name using CrewAI's complete ReAct flow.

The prior override replaced ``call_llm_and_parse`` without preserving the
framework's flow graph. Inheriting the built-in implementation keeps its
``@start``/router/listener wiring and lets ``MockLLM`` read tools from
``from_agent.tools`` without custom function dispatch plumbing.
"""
from crewai.agents.agent_executor import AgentExecutor


class OlaAgentExecutor(AgentExecutor):
    """Use CrewAI's built-in executor flow unchanged."""

