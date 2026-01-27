import contextlib
import logging
import os
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.mcp_server import mcp
from src.agent import Agent, AgentConfig, SparqlResponse
from config.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global agent instance
agent_instance: Agent = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that initializes the MCP session manager and Agent."""
    global agent_instance
    
    # Initialize Agent
    api_key, endpoint, model = settings.get_llm_config()
    
    agent_conf = AgentConfig(
        mcp_server_url=settings.mcp_url,
        provider=settings.llm_provider,
        api_key=api_key,
        endpoint=endpoint, # Can be None for Mistral
        model=model,
        verbose=True,
        enabled_tools=settings.enabled_tools
    )
    agent_instance = Agent(agent_conf)
    
    yield
    
app = FastAPI(
    title="Decide MCP Server",
    description="Decide Project MCP Server Helper",
    version="1.0.0",
    lifespan=lifespan,
)

# Request Models
class QueryRequest(BaseModel):
    query: str

class SparqlRequest(BaseModel):
    entity_class: str
    entity_label: str
    location: str

# Endpoints

@app.post("/agent/query")
async def run_request(request: QueryRequest):
    """Perform a free-form entity linking query via the agent."""
    return {"result": await agent_instance.run_request(request.query)}

@app.post("/agent/query_structured", response_model=SparqlResponse)
async def run_sparql_request_structured(request: SparqlRequest):
    """Perform a structured entity linking via the agent."""
    return await agent_instance.run_sparql_request_structured(
        entity_class=request.entity_class,
        entity_label=request.entity_label,
        location=request.location
    )


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the MCP server
# Adapting from Swiss Sparql-llm github
try:
    if hasattr(mcp, "streamable_http_app"):
        app.mount("/mcp", mcp.streamable_http_app(), name="mcp")
        logger.info("Mounted MCP via streamable_http_app()")
    elif hasattr(mcp, "http_app"):
        # Current FastMCP version uses http_app method
        app.mount("/mcp", mcp.http_app(transport="sse"), name="mcp")
        logger.info("Mounted MCP via http_app(transport='sse')")
    elif hasattr(mcp, "sse_app"):
        # Some versions expose sse_app
        app.mount("/mcp", mcp.sse_app, name="mcp")
        logger.info("Mounted MCP via sse_app")
    elif hasattr(mcp, "_sse_app"):
         app.mount("/mcp", mcp._sse_app, name="mcp")
         logger.info("Mounted MCP via _sse_app")
    else:
        # If it's the raw FastMCP object and we can't find the app method
        # We might need to check if the user meant to use a specific adapter
        logger.warning("Could not find suitable mounting method for MCP object. /mcp endpoint might not be available.")
except Exception as e:
    logger.error(f"Failed to mount MCP app: {e}")

@app.get("/")
async def health():
    return {"status": "running", "endpoints": ["/mcp"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
