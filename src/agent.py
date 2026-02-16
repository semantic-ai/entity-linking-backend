import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

from click import prompt
from fastapi import HTTPException
from fastmcp import Client
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
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
    """Creates a LangChain Tool from an MCP tool definition."""
    schema = getattr(tool_info, "inputSchema", {})
    # Only create model if there are properties, else None (or empty model)
    if schema and "properties" in schema:
        pydantic_model = json_schema_to_pydantic(schema, tool_info.name)
    else:
        pydantic_model = None

    async def _acall(**kwargs):
        if verbose:
            logger.info(f"Invoking tool {tool_info.name} with args: {kwargs}")
        async with client:
            result = await client.call_tool(tool_info.name, kwargs)
            # Extract text content if possible
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

    def _call(**kwargs):
        # Sync wrapper not recommended for async usage but needed for LC sync methods
        return asyncio.run(_acall(**kwargs))

    return StructuredTool.from_function(
        func=_call,
        coroutine=_acall,
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

class AgentConfig(BaseModel):
    mcp_server_url: str
    provider: str = "openai"  # or "mistral"
    api_key: Optional[str] = None # Not needed for Ollama
    endpoint: Optional[str] = None # Can be None for Mistral
    model: str = "gpt-4.1" 
    temperature: float = 0.0
    verbose: bool = False
    enabled_tools: Optional[List[str]] = None

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
                "temperature": self.config.temperature
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

    async def get_tools(self):
        """Returns the list of tools available to the agent."""
        if not hasattr(self, 'lc_tools'):
            await self.initialize()
        return self.lc_tools

    async def initialize(self):
        """Connects to MCP, loads tools, and builds the agent."""
        try:
            logger.info(f"Connecting to MCP at {self.config.mcp_server_url}")
            async with self.mcp_client:
                tools_info = await self.mcp_client.list_tools()
                logger.info(f"Found {len(tools_info)} tools")
            
            # Filter tools if enabled_tools is set
            if self.config.enabled_tools is not None:
                tools_info = [t for t in tools_info if t.name in self.config.enabled_tools]
                logger.info(f"Filtered to {len(tools_info)} tools: {[t.name for t in tools_info]}")

            # Convert to LangChain tools
            self.lc_tools = [create_mcp_tool(t, self.mcp_client, verbose=self.config.verbose) for t in tools_info]
            

            # We use tool calling agent
            self.agent = create_agent(self.llm, tools=self.lc_tools, response_format=SparqlResponse)

            logger.info("Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise e


    async def run_request(self, query: str) -> SparqlResponse:
        """Method to run a general query via the agent and return structured data."""

        if not hasattr(self, 'agent'):
            await self.initialize()

        try:
            # Use ainvoke (async) with asyncio.wait_for to enforce a strict timeout
            result = await asyncio.wait_for(
                self.agent.ainvoke(
                    {"messages": [{"role": "user", "content": query}]}
                ),
                timeout=60.0 # Timeout in seconds
            )
            
            if isinstance(result, dict) and "structured_response" in result:
                return result["structured_response"]
            elif isinstance(result, SparqlResponse):
                return result
            else:
                 # Fallback/Debug
                 logger.warning(f"Unexpected result format: {type(result)}: {result}")
                 # Try to force parse if it's correct type but not in dict
                 if hasattr(result, "structured_response"):
                      return result.structured_response
                 raise ValueError("Could not find structured_response in agent output")

        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Operation timed out after 60 seconds")
        except Exception as e:
            logger.error(f"Error in sparql request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def run_sparql_request_structured(self, entity_class: str, entity_label: str, location: str) -> SparqlResponse:
        """Specific method to run the SPARQL finding task and return structured data based on structured inputs."""
      
        query_template = """Write a SPARQL query to find the URI of the {classification_class} {entity_label} in region {location}, execute it and return the results.
        Keep iterating until you find the best possible match. Provide reasoning for your selection."""
        
        formatted_query = query_template.format(
            classification_class=entity_class, 
            entity_label=entity_label, 
            location=location
        )
        return await self.run_request(formatted_query)