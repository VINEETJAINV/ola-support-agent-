"""
Mock ChatCompletionClient for Autogen that implements deterministic behavior for testing.
Implements all abstract methods of autogen_core.models.ChatCompletionClient.
"""
import os
import json
from typing import Any, Dict, List, Optional, Sequence, Mapping, AsyncGenerator, Union
from dataclasses import dataclass
from datetime import datetime

from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage, ModelInfo, ModelFamily, ModelCapabilities
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel, ConfigDict

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


@dataclass
class MockClientConfig:
    """Configuration for the mock client."""
    responses: List[str]
    structured_output_type: Optional[type] = None


class MockChatCompletionClient(ChatCompletionClient):
    """
    Mock ChatCompletionClient for Autogen that implements deterministic behavior.
    Implements all abstract methods of autogen_core.models.ChatCompletionClient.
    """
    
    def __init__(self, config: MockClientConfig):
        self._config = config
        self._call_count = 0
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self) -> ModelInfo:
        """Return model info with structured_output=True."""
        return ModelInfo(
            function_calling=True,
            json_output=True,
            vision=False,
            family=ModelFamily.UNKNOWN,
            structured_output=True,
        )
    
    @property
    def capabilities(self):
        """Deprecated property - returns model_info."""
        return self.model_info
    
    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Create a single response from the mock model."""
        self._call_count += 1
        
        # Get the next response from the configured responses
        response_index = min(self._call_count - 1, len(self._config.responses) - 1)
        response_content = self._config.responses[response_index]
        
        # If structured output is requested, ensure the response is valid JSON
        if json_output and isinstance(json_output, type):
            # The response should already be valid JSON matching the schema
            pass
        
        # Create a mock CreateResult
        # We need to create a proper CreateResult - let's check what fields it needs
        usage = RequestUsage(prompt_tokens=10, completion_tokens=20)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
        )
        self._actual_usage = usage
        
        return CreateResult(
            content=response_content,
            usage=usage,
            finish_reason="stop",
            cached=False,
        )
    
    async def create_stream(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a stream of responses."""
        result = await self.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result.content
        yield result
    
    async def close(self) -> None:
        """Close the client."""
        pass
    
    def actual_usage(self) -> RequestUsage:
        """Return the usage of the last call."""
        return self._actual_usage
    
    def total_usage(self) -> RequestUsage:
        """Return the total usage across all calls."""
        return self._total_usage
    
    def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = []) -> int:
        """Count tokens in messages."""
        total_chars = sum(len(str(m)) for m in messages)
        return max(1, total_chars // 4)
    
    def remaining_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = []) -> int:
        """Return remaining tokens (mock returns large number)."""
        return 100000


def create_mock_autogen_client(responses: List[str], structured_output_type: Optional[type] = None) -> MockChatCompletionClient:
    """Factory function to create a MockChatCompletionClient."""
    config = MockClientConfig(responses=responses, structured_output_type=structured_output_type)
    return MockChatCompletionClient(config)