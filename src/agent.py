"""Core Agent class — orchestrates LLM, MCP tools, and research workflows.

Delegates to extracted modules:
- src.models        — AgentConfig, SparqlResponse, ResearchResponse
- src.mcp_tools     — MCP-to-LangChain tool conversion
- src.serialization — message tracing, serialization, tool extraction
- src.prompts       — system prompts
- src.logging_callbacks — AgentStepLogger
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from helpers import logger
from src.agent_helpers.mcp_tools import create_mcp_tool
from src.agent_helpers.models import AgentConfig, SparqlResponse, SparqlResult, ResearchResponse
from src.agent_helpers.serialization import json_safe, trace_messages, serialize_agent_message, extract_tool_calls, extract_tool_results
from src.agent_helpers.prompts import RESEARCH_SYSTEM_PROMPT
from src.agent_helpers.logging_callbacks import AgentStepLogger


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
        "step_tool_calls": 0,
        "consecutive_empty_results": 0,
        "same_tool_repeat_count": 0,
        "last_tool_name": "",
        "intervention_count": 0,
        "step_results": [],
        "final_answer": "",
        "error": "",
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


def _extract_final_answer(messages: list) -> str:
    """Walk messages in reverse to find the last non-empty AI response."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                return content
    return ""


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
                request_timeout=self.config.llm_request_timeout,
            )

    # -- Initialization --

    def initialize(self):
        """Connect to MCP, load tools, and build the default agent."""
        logger.info(f"Connecting to MCP at {self.config.mcp_server_url}")

        async def _list_tools():
            async with self.mcp_client:
                return await self.mcp_client.list_tools()

        tools_info = asyncio.run(_list_tools())
        logger.info(f"Found {len(tools_info)} tools")

        if self.config.enabled_tools is not None:
            tools_info = [t for t in tools_info if t.name in self.config.enabled_tools]
            logger.info(f"Filtered to {len(tools_info)} tools: {[t.name for t in tools_info]}")

        self.lc_tools = [create_mcp_tool(t, self.mcp_client, verbose=self.config.verbose) for t in tools_info]
        self.cached_tools = {t.name: t for t in self.lc_tools}
        self.agent = create_agent(self.llm, tools=self.lc_tools, response_format=SparqlResponse)
        logger.info("Agent initialized successfully")

    def _ensure_initialized(self):
        if not hasattr(self, "cached_tools"):
            self.initialize()

    def get_tools(self):
        """Return the list of LangChain tools available to the agent."""
        self._ensure_initialized()
        return self.lc_tools

    def _select_tools(self, names: Optional[List[str]] = None):
        """Return a tool list — filtered by *names* if given, else all."""
        self._ensure_initialized()
        if names is not None:
            return [self.cached_tools[n] for n in names if n in self.cached_tools]
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
            max_interventions=self.config.max_interventions_per_step,
            max_step_tool_calls=self.config.max_step_tool_calls,
            max_replans=self.config.max_replans,
        )

    # -----------------------------------------------------------------------
    # Structured SPARQL requests
    # -----------------------------------------------------------------------

    def run_request(self, query: str) -> SparqlResponse:
        """Run a general query and return structured SparqlResponse."""
        return self._run_request(query)

    def run_sparql_request_structured(self, entity_class: str, entity_label: str, location: str = "N/A") -> SparqlResponse:
        """Run a SPARQL entity-finding task with class-specific config."""
        query_template = (
            "Write a SPARQL query to find the URI of the {classification_class} "
            "{entity_label} in region {location}, execute it and return the results. "
            "Keep iterating until you find the best possible match. Provide reasoning for your selection."
        )
        specific_tools = None

        if self.config.entity_class_configs:
            class_key = entity_class.strip().lower()
            mapping = {k.strip().lower(): v for k, v in self.config.entity_class_configs.items()}
            if class_key in mapping:
                conf = mapping[class_key]
                logger.info(f"Found specific configuration for class '{class_key}': {conf}")
                specific_tools = conf.get("tools")
                query_template = conf.get("query_template", query_template)
            else:
                logger.warning(
                    f"No entity_class_configs entry for '{entity_class}' "
                    f"(normalized: '{class_key}'). Known keys: {sorted(mapping.keys())}. "
                    f"Falling back to default template."
                )

        formatted_query = query_template.format(
            classification_class=entity_class,
            entity_label=entity_label,
            location=location,
        )
        return self._run_request(formatted_query, specific_tools=specific_tools)

    def _run_request(self, query: str, specific_tools: Optional[List[str]] = None) -> SparqlResponse:
        tools_to_use = self._select_tools(specific_tools)
        if not tools_to_use:
            logger.warning("No tools available for this request.")
        logger.info(f"[RUN] Tools used: {[t.name for t in tools_to_use]}")

        temp_agent = create_agent(self.llm, tools=tools_to_use, response_format=SparqlResponse)

        try:
            result = temp_agent.invoke({"messages": [{"role": "user", "content": query}]})
            messages = result.get("messages", []) if isinstance(result, dict) else []

            if messages and self.config.tracing_enabled:
                trace_messages(messages)

            if isinstance(result, dict) and "structured_response" in result:
                return result["structured_response"]
            if isinstance(result, SparqlResponse):
                return result
            if hasattr(result, "structured_response"):
                return result.structured_response

            error_msg = "Could not find structured_response in agent output."
            if messages:
                content = getattr(messages[-1], "content", "")
                if content:
                    error_msg += f"\n\nLast agent response:\n{content}"
            raise ValueError(error_msg)

        except HTTPException:
            raise
        except Exception as e:
            _handle_llm_error(e, "sparql request")

    # -----------------------------------------------------------------------
    # Research requests (free-form)
    # -----------------------------------------------------------------------

    def run_research_request(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        """Public entry point — delegates to graph or flat agent based on config."""
        if self.config.planning_enabled:
            if self.config.streaming_enabled:
                return self.stream_research_request(query, messages=messages)
            else:
                return self._run_research_graph(query, messages=messages)
        else:
            return self._run_research_flat(query, messages=messages)

    def _run_research_flat(
        self,
        query: str,
        specific_tools: Optional[List[str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        """Run a free-form research query using a flat ReAct agent (no planning)."""
        tools_to_use = self._select_tools(specific_tools)
        if not tools_to_use:
            logger.warning("No tools available for this request.")
        logger.info(f"[RESEARCH] Tools used: {[t.name for t in tools_to_use]}")

        temp_agent = create_agent(self.llm, tools=tools_to_use, system_prompt=RESEARCH_SYSTEM_PROMPT)
        step_logger = AgentStepLogger()
        input_messages = _sanitize_messages(query, messages)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    temp_agent.invoke,
                    {"messages": input_messages},
                    {"recursion_limit": self.config.research_recursion_limit, "callbacks": [step_logger]},
                )
                try:
                    result = future.result(timeout=self.config.research_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"[RESEARCH] Agent timed out after {self.config.research_timeout}s")
                    raise HTTPException(
                        status_code=504,
                        detail=f"Research request timed out after {self.config.research_timeout} seconds.",
                    )

            result_messages = result.get("messages", []) if isinstance(result, dict) else []
            trace = trace_messages(result_messages, self.config.tracing_enabled) if result_messages else None
            serialized = [serialize_agent_message(m, i) for i, m in enumerate(result_messages, 1)]
            tool_calls_list = extract_tool_calls(serialized)
            tool_results_list = extract_tool_results(serialized)

            answer = _extract_final_answer(result_messages)

            # Backwards-compatible sparql_results from tool output
            sparql_results = [
                {"tool": r.get("tool", "unknown"), "tool_call_id": r.get("tool_call_id"), "result": r.get("content", "")}
                for r in tool_results_list
            ] or None

            raw_response = None
            if isinstance(result, dict):
                extra = {k: json_safe(v) for k, v in result.items() if k != "messages"}
                raw_response = extra or None

            return ResearchResponse(
                answer=answer,
                sources=None,
                sparql_results=sparql_results,
                messages=serialized or None,
                tool_calls=tool_calls_list or None,
                tool_results=tool_results_list or None,
                trace=trace,
                raw_response=raw_response,
            )

        except HTTPException:
            raise
        except Exception as e:
            _handle_llm_error(e, "research request")

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
