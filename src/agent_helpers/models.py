"""Pydantic models for agent configuration, requests, and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The answer or research findings")
    sources: Optional[List[str]] = Field(None, description="Sources or URIs referenced")
    sparql_results: Optional[List[Dict[str, Any]]] = Field(None, description="Raw SPARQL query results if any were executed")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Full serialized LangChain message sequence")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls requested by the model")
    tool_results: Optional[List[Dict[str, Any]]] = Field(None, description="Tool result messages returned to the model")
    trace: Optional[str] = Field(None, description="Human-readable execution trace")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Additional serializable fields returned by the agent runtime")


# --- Agent Configuration ---

class AgentConfig(BaseModel):
    mcp_server_url: str
    provider: str = "mistralai"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: str = "mistral-small"
    temperature: float = 0.0
    verbose: bool = False
    agent_enabled_tools: Optional[List[str]] = None
    llm_max_retries: int = 3
    llm_request_timeout: int = 60

    # Research settings
    research_timeout: int = 600
    retrieve_tools: List[str] = ["search_sparql_docs"]
    execute_tools: Optional[List[str]] = None
    step_timeout_s: int = 90
    max_step_tool_calls: int = 6
    max_replans: int = 2
