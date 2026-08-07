import contextlib
import json
from typing import AsyncIterator, Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, APIRouter, HTTPException
from src.agent import Agent
from src.agent_helpers.models import ResearchResponse
from fastapi.responses import StreamingResponse

from src.mcp_server import mcp

# Setup logging
from helpers import logger
from src.utils.utils import initialize_agent

# Global agent instance
agent_instance: Agent = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that initializes the MCP session manager and Agent."""
    global agent_instance

    agent_instance = initialize_agent()

    yield

class ResearchRequest(BaseModel):
    query: str
    messages: Optional[List[Dict[str, str]]] = None
    
# Initialize the router
router = APIRouter()

# Endpoints

@router.post("/agent/research", response_model=ResearchResponse)
def run_research_request(request: ResearchRequest):
    """Perform a free-form research query via the agent."""
    logger.info(f"Received research query: {request.query}")
    if not agent_instance:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        return agent_instance.run_research_request(request.query, messages=request.messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in research request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research request failed: {str(e)}. Check server logs for the full trace.",
        )


@router.post("/agent/research/stream")
def stream_research_request(request: ResearchRequest):
    """Stream research progress as Server-Sent Events (SSE).

    Each event is a JSON line: {"event": "...", "data": {...}}
    Events: graph, retrieve, plan, step_done, replan, validate, done, error
    """
    logger.info(f"Received streaming research query: {request.query}")
    if not agent_instance:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    def event_generator():
        try:
            for event in agent_instance.stream_research_request(
                request.query, messages=request.messages
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming research: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'detail': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    
@router.get("/")
async def health():
    return {
        "status": "running",
        "endpoints": ["/mcp/sse", "/agent/research", "/agent/research/stream"],
    }

def mount_mcp(app: FastAPI):
    """Mount the pinned FastMCP SSE application below ``/mcp``."""
    try:
        app.mount("/mcp", mcp.http_app(transport="sse"), name="mcp")
        logger.info("Mounted MCP via FastMCP SSE transport at /mcp/sse")
    except Exception as e:
        logger.error(f"Failed to mount MCP app: {e}")
        raise
