"""Core Agent class — orchestrates LLM, MCP tools, and research workflows.

The Agent loads the configured MCP tools and exposes synchronous and streaming
entry points for the LangGraph research workflow.
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from helpers import logger
from src.agent_helpers.mcp_tools import create_mcp_tool
from src.agent_helpers.models import AgentConfig, ResearchResponse
from src.agent_helpers.serialization import json_safe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handle_llm_error(e: Exception, context: str) -> None:
    """Raise the appropriate HTTPException for LLM/agent errors."""
    logger.error(f"Error in {context}: {e}")
    if hasattr(e, "response") and hasattr(e.response, "status_code") and e.response.status_code == 429:
        logger.warning("Rate limited by LLM service")
        raise HTTPException(status_code=429, detail="Upstream rate limit exceeded. Please retry later.")
    raise HTTPException(status_code=500, detail=str(e))


def _build_initial_graph_state(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Return the initial state dict expected by the research graph."""
    return {
        "query": query,
        "messages": messages,
        "retrieved_docs": "",
        "plan": [],
        "current_step_index": 0,
        "replan_count": 0,
        "step_start_time": 0.0,
        "step_results": [],
        "final_answer": "",
    }


def _sanitize_messages(query: str, messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    """Build a clean message list from optional conversation history."""
    if not messages:
        return [{"role": "user", "content": query}]

    allowed_roles = {"system", "user", "assistant"}
    sanitized = [
        {"role": role, "content": str(m.get("content", ""))}
        for m in messages
        if (role := str(m.get("role", "")).lower()) in allowed_roles and m.get("content") is not None
    ]

    if not sanitized or sanitized[-1] != {"role": "user", "content": query}:
        sanitized.append({"role": "user", "content": query})

    return sanitized


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.mcp_client = Client(config.mcp_server_url)
        self.llm = self._create_llm()

    # -- LLM factory --

    def _create_llm(self):
        provider = self.config.provider.lower()
        if provider == "mistral":
            kwargs = {
                "model": self.config.model,
                "api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_retries": self.config.llm_max_retries,
                "timeout": self.config.llm_request_timeout,
            }
            if self.config.endpoint:
                kwargs["base_url"] = self.config.endpoint
            return ChatMistralAI(**kwargs)
        elif provider == "ollama":
            return ChatOllama(
                model=self.config.model,
                base_url=self.config.endpoint,
                temperature=self.config.temperature,
                timeout=self.config.llm_request_timeout,
            )
        else:
            return ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.endpoint,
                temperature=self.config.temperature,
                max_retries=self.config.llm_max_retries,
                request_timeout=self.config.llm_request_timeout,
            )


    # -- Initialization --

    def initialize(self):
        """Connect to MCP and load the tools available to the research graph."""
        logger.info(f"Connecting to MCP at {self.config.mcp_server_url}")

        async def _list_tools():
            async with self.mcp_client:
                return await self.mcp_client.list_tools()

        tools_info = asyncio.run(_list_tools())
        logger.info(f"Found {len(tools_info)} tools")

        if self.config.agent_enabled_tools is not None:
            tools_info = [t for t in tools_info if t.name in self.config.agent_enabled_tools]
            logger.info(f"Filtered to {len(tools_info)} tools: {[t.name for t in tools_info]}")

        self.lc_tools = [create_mcp_tool(t, self.mcp_client, verbose=self.config.verbose) for t in tools_info]
        logger.info("Agent initialized successfully")
        
    def _ensure_initialized(self):
        if not hasattr(self, "lc_tools"):
            self.initialize()

    def get_tools(self):
        """Return the list of LangChain tools available to the agent."""
        self._ensure_initialized()
        return self.lc_tools

    # -- Research graph factory (lazy import) --

    def _build_research_graph(self):
        from src.agent_helpers.research_graph import create_research_graph
        return create_research_graph(
            llm=self.llm,
            tools=self.lc_tools,
            retrieve_tool_names=self.config.retrieve_tools,
            execute_tool_names=self.config.execute_tools,
            step_timeout_s=self.config.step_timeout_s,
            max_step_tool_calls=self.config.max_step_tool_calls,
            max_replans=self.config.max_replans,
        )

    # -----------------------------------------------------------------------
    # Structured SPARQL requests
    # -----------------------------------------------------------------------

    def run_research_request(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        """Run the non-streaming plan-execute graph and return one response."""
        return self._run_research_graph(query, messages=messages)

    # -----------------------------------------------------------------------
    # Research requests (plan-execute graph)
    # -----------------------------------------------------------------------

    def _run_research_graph(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        """Run a research query using the LangGraph plan-execute architecture."""
        self._ensure_initialized()

        logger.info("[RESEARCH-GRAPH] Building plan-execute graph...")
        graph = self._build_research_graph()

        try:
            mermaid = graph.get_graph().draw_mermaid()
            logger.info(f"[RESEARCH-GRAPH] Graph structure:\n{mermaid}")
        except Exception:
            pass

        initial_state = _build_initial_graph_state(query, _sanitize_messages(query, messages))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(graph.invoke, initial_state)
                try:
                    result = future.result(timeout=self.config.research_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"[RESEARCH-GRAPH] Timed out after {self.config.research_timeout}s")
                    raise HTTPException(
                        status_code=504,
                        detail=f"Research request timed out after {self.config.research_timeout} seconds.",
                    )

            return ResearchResponse(
                answer=result.get("final_answer", ""),
                sources=None,
                sparql_results=None,
                messages=None,
                tool_calls=None,
                tool_results=None,
                trace=None,
                raw_response={
                    "plan": json_safe(result.get("plan", [])),
                    "step_results": json_safe(result.get("step_results", [])),
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            _handle_llm_error(e, "research graph")

    # -----------------------------------------------------------------------
    # Streaming research (SSE)
    # -----------------------------------------------------------------------

    def stream_research_request(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        """Generator yielding progress events as the research graph executes.

        Yields dicts: {"event": <type>, "data": <payload>}
        Event types: graph, retrieve, plan, step_done, replan, validate, done, error
        """
        self._ensure_initialized()
        graph = self._build_research_graph()

        try:
            mermaid = graph.get_graph().draw_mermaid()
            yield {"event": "graph", "data": {"mermaid": mermaid}}
        except Exception:
            pass

        initial_state = _build_initial_graph_state(query, _sanitize_messages(query, messages))

        try:
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in event.items():
                    if node_name == "retrieve":
                        yield {"event": "retrieve", "data": {"docs_length": len(update.get("retrieved_docs", ""))}}
                    elif node_name == "plan":
                        yield {"event": "plan", "data": {"plan": json_safe(update.get("plan", []))}}
                    elif node_name == "execute":
                        step_results = update.get("step_results", [])
                        yield {
                            "event": "step_done",
                            "data": {
                                "current_step_index": update.get("current_step_index", 0),
                                "step_result": json_safe(step_results[-1]) if step_results else None,
                            },
                        }
                    elif node_name == "replan":
                        yield {"event": "replan", "data": {"plan": json_safe(update.get("plan", []))}}
                    elif node_name == "validate":
                        yield {"event": "validate", "data": {"answer": update.get("final_answer", "")}}

            yield {"event": "done", "data": {}}

        except Exception as e:
            logger.error(f"Error in research graph stream: {e}")
            yield {"event": "error", "data": {"detail": str(e)}}
