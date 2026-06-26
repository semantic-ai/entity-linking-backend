"""Helpers to convert MCP tool definitions into sync LangChain StructuredTools."""

import asyncio
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from helpers import logger


def json_schema_to_pydantic(schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
    """Convert a JSON schema to a Pydantic model dynamically."""
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

        if json_type == "array":
            items = prop.get("items", {})
            item_type_str = items.get("type")
            if isinstance(item_type_str, list):
                item_type_str = item_type_str[0]
            item_type = type_map.get(item_type_str, Any)
            p_type = List[item_type]
        else:
            p_type = type_map.get(json_type, Any)

        if name in required:
            field_def = Field(..., description=description)
        else:
            field_def = Field(None, description=description)
            p_type = Optional[p_type]

        fields[name] = (p_type, field_def)

    return create_model(model_name, **fields)


def create_mcp_tool(tool_info, client, verbose: bool = False, tool_timeout: int = 60) -> StructuredTool:
    """Create a sync LangChain Tool from an MCP tool definition.

    FastMCP's Client is async-only, so calls are dispatched via asyncio.run.
    A fresh Client is created per call to avoid cross-thread contention on
    a shared connection, and the entire operation (connect + call) is wrapped
    in a single timeout.
    """
    from fastmcp import Client

    # Extract the server URL from the client so each call can create its own connection
    if hasattr(client, "transport") and hasattr(client.transport, "url"):
        server_url = client.transport.url
    else:
        server_url = str(client)

    schema = getattr(tool_info, "inputSchema", {})
    pydantic_model = (
        json_schema_to_pydantic(schema, tool_info.name)
        if schema and "properties" in schema
        else None
    )

    def _call(**kwargs):
        logger.info(f"[TOOL-CALL] {tool_info.name} called with args: {kwargs}")

        async def _run_async():
            # Create a fresh client per call to avoid shared-state issues
            call_client = Client(server_url)
            async with call_client:
                result = await call_client.call_tool(tool_info.name, kwargs)
                if hasattr(result, "content") and isinstance(result.content, list):
                    text_parts = [
                        item.text if hasattr(item, "text") else item["text"]
                        for item in result.content
                        if hasattr(item, "text") or (isinstance(item, dict) and "text" in item)
                    ]
                    if text_parts:
                        return "\n".join(text_parts)
                return str(result)

        async def _run_with_timeout():
            # Wrap the entire operation (connect + call) in a single timeout
            return await asyncio.wait_for(_run_async(), timeout=tool_timeout)

        import time as _time
        _start = _time.time()
        try:
            result = asyncio.run(_run_with_timeout())
            logger.info(f"[TOOL-DONE] {tool_info.name} completed in {_time.time() - _start:.1f}s")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[TOOL-TIMEOUT] {tool_info.name} timed out after {tool_timeout}s")
            return f"Tool '{tool_info.name}' timed out after {tool_timeout}s. The endpoint may be unresponsive. Try a different approach."
        except Exception as e:
            logger.error(f"[TOOL-ERROR] {tool_info.name} failed after {_time.time() - _start:.1f}s: {e}")
            return f"Tool '{tool_info.name}' failed: {str(e)}. Try a different approach."

    return StructuredTool.from_function(
        func=_call,
        name=tool_info.name,
        description=tool_info.description or f"Tool {tool_info.name}",
        args_schema=pydantic_model,
    )
