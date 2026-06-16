import asyncio
from typing import Any, Dict, List, Optional, Type, Union

from click import prompt
from fastapi import HTTPException
from fastmcp import Client
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, create_model

# Clean logging
from helpers import logger


# --- Helper Functions ---

def json_schema_to_pydantic(schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
    """Helper to convert a JSON schema to a Pydantic model dynamically."""
    fields = {}
    if not schema or "properties" not in schema:
        return create_model(model_name)
    
    required = set(schema.get("required", []))
    
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }

    for name, prop in schema.get("properties", {}).items():
        json_type = prop.get("type")
        
        # Handle anyOf (e.g. for Optional/Nullable types)
        if not json_type and "anyOf" in prop:
             for option in prop["anyOf"]:
                  t = option.get("type")
                  if t and t != "null":
                       json_type = t
                       break

        if isinstance(json_type, list):
             json_type = json_type[0]
             
        description = prop.get("description", "")
        
        # Handle arrays
        if json_type == "array":
            items = prop.get("items", {})
            item_type_str = items.get("type")
            if isinstance(item_type_str, list):
                item_type_str = item_type_str[0]
            
            # recursive or simple? assuming simple for tools
            # Python < 3.9 List[T], >= 3.9 list[T]. usage of List from typing is safer compatibility
            item_type = type_map.get(item_type_str, Any)
            p_type = List[item_type]
        else:
            p_type = type_map.get(json_type, Any)
        
        # Helper to determine if nullable
        # (Simplified logic, assumes if not required it is optional)
        if name in required:
             # Field(..., ...) means required
            field_def = Field(..., description=description)
        else:
            field_def = Field(None, description=description)
            p_type = Optional[p_type]
            
        fields[name] = (p_type, field_def)
        
    return create_model(model_name, **fields)

def create_mcp_tool(tool_info, client, verbose: bool = False):
    """Creates a sync LangChain Tool from an MCP tool definition.

    FastMCP's Client is async-only, so the call is dispatched through
    asyncio.run on a local async helper. The wrapper itself is sync;
    callers must invoke it from a sync context (no running event loop).
    """
    schema = getattr(tool_info, "inputSchema", {})
    # Only create model if there are properties, else None (or empty model)
    if schema and "properties" in schema:
        pydantic_model = json_schema_to_pydantic(schema, tool_info.name)
    else:
        pydantic_model = None

    def _call(**kwargs):
        if verbose:
            logger.info(f"Invoking tool {tool_info.name} with args: {kwargs}")

        async def _run_async():
            async with client:
                result = await client.call_tool(tool_info.name, kwargs)
                if hasattr(result, "content") and isinstance(result.content, list):
                    text_content = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            text_content.append(item.text)
                        elif isinstance(item, dict) and "text" in item:
                            text_content.append(item["text"])
                    if text_content:
                        return "\n".join(text_content)
                return str(result)

        return asyncio.run(_run_async())

    return StructuredTool.from_function(
        func=_call,
        name=tool_info.name,
        description=tool_info.description or f"Tool {tool_info.name}",
        args_schema=pydantic_model
    )

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

class AgentConfig(BaseModel):
    mcp_server_url: str
    provider: str = "mistralai"  # Options: "mistral", "ollama", "openai"
    api_key: Optional[str] = None # Not needed for Ollama
    endpoint: Optional[str] = None # Can be None for Mistral
    model: str = "mistral-small" 
    temperature: float = 0.0
    verbose: bool = False
    tracing_enabled: bool = False
    enabled_tools: Optional[List[str]] = None
    entity_class_configs: Optional[Dict[str, Any]] = None
    llm_max_retries: int = 3

# --- Agent Class ---

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.mcp_client = Client(config.mcp_server_url)
        
        # Initialize LLM
        if self.config.provider.lower() == "mistral":
            kwargs = {
                "model": self.config.model,
                "api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_retries": self.config.llm_max_retries
            }
            if self.config.endpoint:
                kwargs["base_url"] = self.config.endpoint
                
            self.llm = ChatMistralAI(**kwargs)
        elif self.config.provider.lower() == "ollama":
            self.llm = ChatOllama(
                model=self.config.model,
                base_url=self.config.endpoint, # Maps to ollama_url
                temperature=self.config.temperature
            )
        else:   
            self.llm = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.endpoint,
                temperature=self.config.temperature
            )

    def get_tools(self):
        """Returns the list of tools available to the agent."""
        if not hasattr(self, 'lc_tools'):
            self.initialize()
        return self.lc_tools

    def initialize(self):
        """Connects to MCP, loads tools, and builds the agent."""
        try:
            logger.info(f"Connecting to MCP at {self.config.mcp_server_url}")

            async def _list_tools_async():
                async with self.mcp_client:
                    return await self.mcp_client.list_tools()

            tools_info = asyncio.run(_list_tools_async())
            logger.info(f"Found {len(tools_info)} tools")

            # Filter tools if enabled_tools is set
            if self.config.enabled_tools is not None:
                tools_info = [t for t in tools_info if t.name in self.config.enabled_tools]
                logger.info(f"Filtered to {len(tools_info)} tools: {[t.name for t in tools_info]}")

            # Convert to LangChain tools
            self.lc_tools = [create_mcp_tool(t, self.mcp_client, verbose=self.config.verbose) for t in tools_info]
            self.cached_tools = {t.name: t for t in self.lc_tools}

            # We use tool calling agent
            self.agent = create_agent(self.llm, tools=self.lc_tools, response_format=SparqlResponse)

            logger.info("Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise e

    @staticmethod
    def trace_messages(messages: list, log: bool = True) -> str:
        """Format the full sequence of agent messages/tool calls for debugging.

        Returns a human-readable trace string and optionally logs it.
        """
        lines: List[str] = []
        step = 0

        for msg in messages:
            msg_type = type(msg).__name__

            if isinstance(msg, HumanMessage):
                step += 1
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                lines.append(f"\n{'='*80}")
                lines.append(f"Step {step} | HUMAN")
                lines.append(f"{'='*80}")
                lines.append(content)

            elif isinstance(msg, AIMessage):
                step += 1
                lines.append(f"\n{'='*80}")
                lines.append(f"Step {step} | AI")
                lines.append(f"{'='*80}")

                # Show text content (if any)
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    lines.append(f"Content: {content}")

                # Show tool calls
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    lines.append(f"Tool calls ({len(tool_calls)}):")
                    for tc in tool_calls:
                        name = tc.get("name", tc.get("function", {}).get("name", "?"))
                        args = tc.get("args", {})
                        tc_id = tc.get("id", "")
                        lines.append(f"  -> {name}({args})  [id={tc_id}]")

            elif isinstance(msg, ToolMessage):
                step += 1
                tool_name = getattr(msg, "name", "unknown")
                tc_id = getattr(msg, "tool_call_id", "")
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                lines.append(f"\n{'-'*80}")
                lines.append(f"Step {step} | TOOL RESULT: {tool_name}  [id={tc_id}]")
                lines.append(f"{'-'*80}")
                lines.append(content)

            else:
                step += 1
                lines.append(f"\nStep {step} | {msg_type}: {str(msg)}")

        trace = "\n".join(lines)
        if log:
            logger.info(f"\n{'#'*80}\n# AGENT TRACE\n{'#'*80}{trace}\n{'#'*80}\n# END TRACE\n{'#'*80}")
        return trace

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert LangChain/Pydantic objects to JSON-safe structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, BaseModel):
            try:
                return value.model_dump(mode="json")
            except Exception:
                return Agent._json_safe(value.model_dump())

        if isinstance(value, dict):
            return {str(k): Agent._json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [Agent._json_safe(v) for v in value]

        if hasattr(value, "model_dump"):
            try:
                return Agent._json_safe(value.model_dump(mode="json"))
            except Exception:
                try:
                    return Agent._json_safe(value.model_dump())
                except Exception:
                    pass

        return str(value)

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, list, tuple, set, dict)):
            return bool(value)
        return True

    @classmethod
    def serialize_agent_message(cls, msg: Any, index: int) -> Dict[str, Any]:
        """Serialize a LangChain message without dropping provider/tool metadata."""
        serialized: Dict[str, Any] = {
            "index": index,
            "type": type(msg).__name__,
            "role": getattr(msg, "type", type(msg).__name__.replace("Message", "").lower()),
        }

        for attr in (
            "id",
            "name",
            "content",
            "additional_kwargs",
            "response_metadata",
            "usage_metadata",
            "tool_calls",
            "invalid_tool_calls",
            "tool_call_id",
            "status",
            "artifact",
        ):
            if hasattr(msg, attr):
                value = getattr(msg, attr)
                if attr == "content" or cls._has_value(value):
                    serialized[attr] = cls._json_safe(value)

        return serialized

    @classmethod
    def extract_tool_calls(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten normalized and provider-native tool calls from serialized messages."""
        tool_calls: List[Dict[str, Any]] = []

        for message in messages:
            normalized_calls = message.get("tool_calls") or []
            for call_index, tool_call in enumerate(normalized_calls):
                call = cls._json_safe(tool_call)
                if not isinstance(call, dict):
                    call = {"value": call}
                call["message_index"] = message.get("index")
                call["call_index"] = call_index
                call["source"] = "normalized"
                tool_calls.append(call)

            if normalized_calls:
                continue

            additional_kwargs = message.get("additional_kwargs") or {}
            if not isinstance(additional_kwargs, dict):
                continue

            raw_calls = additional_kwargs.get("tool_calls", [])
            for call_index, tool_call in enumerate(raw_calls):
                call = cls._json_safe(tool_call)
                if not isinstance(call, dict):
                    call = {"value": call}
                call["message_index"] = message.get("index")
                call["call_index"] = call_index
                call["source"] = "provider"
                tool_calls.append(call)

        return tool_calls

    @classmethod
    def extract_tool_results(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten tool result messages from the serialized message stream."""
        tool_results: List[Dict[str, Any]] = []

        for message in messages:
            if message.get("type") != "ToolMessage" and message.get("role") != "tool":
                continue

            result: Dict[str, Any] = {
                "message_index": message.get("index"),
                "tool": message.get("name", "unknown"),
                "tool_call_id": message.get("tool_call_id"),
                "status": message.get("status"),
                "content": message.get("content", ""),
            }
            if "artifact" in message:
                result["artifact"] = message["artifact"]

            tool_results.append(result)

        return tool_results

    def _run_request(self, query: str, specific_tools: Optional[List[str]] = None) -> SparqlResponse:
        """Internal method to run a query with optionally specific tools JIT."""
        if not hasattr(self, 'cached_tools'):
            self.initialize()

        # Determine which tools to use for this request
        if specific_tools is not None:
            tools_to_use = [self.cached_tools[name] for name in specific_tools if name in self.cached_tools]
        else:
            tools_to_use = self.lc_tools

        if not tools_to_use:
            logger.warning("No tools available for this request.")
      
        logger.info(f"[RUN] Tools used: {[t.name for t in tools_to_use]}")

        # JIT Agent Creation (lightweight operation)
        temp_agent = create_agent(self.llm, tools=tools_to_use, response_format=SparqlResponse)

        try:
            result = temp_agent.invoke({"messages": [{"role": "user", "content": query}]})

            # Always trace the full message sequence for debugging
            messages = result.get("messages", []) if isinstance(result, dict) else []

            if messages and self.config.tracing_enabled:
                self.trace_messages(messages)

            if isinstance(result, dict) and "structured_response" in result:
                return result["structured_response"]
            elif isinstance(result, SparqlResponse):
                return result
            else:
                # Fallback/Debug
                logger.warning(f"Unexpected result format: {type(result)}")

                # Try to force parse if it's correct type but not in dict
                if hasattr(result, "structured_response"):
                    return result.structured_response

                error_msg = "Could not find structured_response in agent output."
                if messages:
                    last_msg = messages[-1]
                    content = getattr(last_msg, "content", "")
                    if content:
                        error_msg += f"\n\n======================Last agent response ========================\n\n {content}"

                raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"Error in sparql request: {e}")
            if hasattr(e, "response") and hasattr(e.response, "status_code") and e.response.status_code == 429:
                logger.warning("Rate limited by LLM service")
                raise HTTPException(
                    status_code=429,
                    detail="Upstream rate limit exceeded. Please retry later.",
                )
            else:
                raise HTTPException(status_code=500, detail=str(e))

    def run_request(self, query: str) -> SparqlResponse:
        """Method to run a general query via the agent and return structured data (uses all enabled tools)."""
        return self._run_request(query, specific_tools=None)

    def run_sparql_request_structured(self, entity_class: str, entity_label: str, location: str = "N/A") -> SparqlResponse:
        """Specific method to run the SPARQL finding task and return structured data based on structured inputs."""

        # Default fallback template
        query_template = """Write a SPARQL query to find the URI of the {classification_class} {entity_label} in region {location}, execute it and return the results.
        Keep iterating until you find the best possible match. Provide reasoning for your selection."""

        # Determine specific tools and query based on entity class mapping
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
            location=location
        )

        return self._run_request(formatted_query, specific_tools=specific_tools)

    @staticmethod
    def _research_messages(query: str, messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "user", "content": query}]

        allowed_roles = {"system", "user", "assistant"}
        sanitized_messages: List[Dict[str, str]] = []

        for message in messages:
            role = str(message.get("role", "")).lower()
            content = message.get("content", "")
            if role in allowed_roles and content is not None:
                sanitized_messages.append({"role": role, "content": str(content)})

        if not sanitized_messages or sanitized_messages[-1] != {"role": "user", "content": query}:
            sanitized_messages.append({"role": "user", "content": query})

        return sanitized_messages

    def _run_research_request(
        self,
        query: str,
        specific_tools: Optional[List[str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        
        """Internal method to run a free-form research query without enforcing SparqlResponse."""
        if not hasattr(self, 'cached_tools'):
            self.initialize()

        if specific_tools is not None:
            tools_to_use = [self.cached_tools[name] for name in specific_tools if name in self.cached_tools]
        else:
            tools_to_use = self.lc_tools

        if not tools_to_use:
            logger.warning("No tools available for this request.")

        logger.info(f"[RESEARCH] Tools used: {[t.name for t in tools_to_use]}")
        # JIT Agent without response_format — allows free-form output
        temp_agent = create_agent(self.llm, tools=tools_to_use)

        try:
            result = temp_agent.invoke({"messages": self._research_messages(query, messages)})

            messages = result.get("messages", []) if isinstance(result, dict) else []
            trace = self.trace_messages(messages, self.config.tracing_enabled) if messages else None
            serialized_messages = [
                self.serialize_agent_message(msg, index)
                for index, msg in enumerate(messages, start=1)
            ]
            tool_calls = self.extract_tool_calls(serialized_messages)
            tool_results = self.extract_tool_results(serialized_messages)

            # Extract the final AI message content
            answer = ""
            sources = []

            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content.strip():
                        answer = content
                        break

            if not answer:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

            raw_response = None
            if isinstance(result, dict):
                raw_response = {}
                for key, value in result.items():
                    if key == "messages":
                        continue
                    raw_response[key] = self._json_safe(value)
                if not raw_response:
                    raw_response = None

            # Keep the existing sparql_results field populated with tool output for
            # backwards compatibility with clients that already read it.
            sparql_results = [
                {
                    "tool": item.get("tool", "unknown"),
                    "tool_call_id": item.get("tool_call_id"),
                    "result": item.get("content", ""),
                }
                for item in tool_results
            ]

            return ResearchResponse(
                answer=answer,
                sources=sources if sources else None,
                sparql_results=sparql_results if sparql_results else None,
                messages=serialized_messages if serialized_messages else None,
                tool_calls=tool_calls if tool_calls else None,
                tool_results=tool_results if tool_results else None,
                trace=trace,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error in research request: {e}")
            if hasattr(e, "response") and hasattr(e.response, "status_code") and e.response.status_code == 429:
                logger.warning("Rate limited by LLM service")
                raise HTTPException(
                    status_code=429,
                    detail="Upstream rate limit exceeded. Please retry later.",
                )
            else:
                raise HTTPException(status_code=500, detail=str(e))

    def run_research_request(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResponse:
        """Run a free-form research query and return unstructured results."""
        return self._run_research_request(query, specific_tools=None, messages=messages)
