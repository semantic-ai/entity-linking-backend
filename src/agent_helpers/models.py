"""Pydantic models for agent configuration, requests, and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Response Models ---

class SparqlResult(BaseModel):
    uri: str = Field(..., description="The URI of the entity")
    label: str = Field(..., description="The label of the entity")
    location: Optional[str] = Field(None, description="The location associated with the entity.")
    reasoning: str = Field(..., description="The reasoning behind the selection.")


class SparqlResponse(BaseModel):
    results: List[SparqlResult] = Field(..., description="The list of matching entities found")


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
    tracing_enabled: bool = False
    enabled_tools: Optional[List[str]] = None
    entity_class_configs: Optional[Dict[str, Any]] = None
    llm_max_retries: int = 3
    llm_request_timeout: int = 60

    # Research settings
    research_timeout: int = 600
    research_recursion_limit: int = 10

    # Plan-execute graph settings
    planning_enabled: bool = True
    streaming_enabled: bool = True

    retrieve_tools: List[str] = ["search_sparql_docs"]
    execute_tools: Optional[List[str]] = None
    step_timeout_s: int = 90
    max_interventions_per_step: int = 2
    max_step_tool_calls: int = 6
    max_replans: int = 2
