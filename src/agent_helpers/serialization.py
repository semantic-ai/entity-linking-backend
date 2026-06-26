"""Message serialization, tracing, and tool-call extraction utilities."""

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from helpers import logger


def json_safe(value: Any) -> Any:
    """Convert LangChain/Pydantic objects to JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except Exception:
            try:
                return json_safe(value.model_dump())
            except Exception:
                pass
    return str(value)


def trace_messages(messages: list, log: bool = True) -> str:
    """Format the full sequence of agent messages/tool calls for debugging."""
    lines: List[str] = []
    step = 0

    for msg in messages:
        step += 1

        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"\n{'='*80}\nStep {step} | HUMAN\n{'='*80}\n{content}")

        elif isinstance(msg, AIMessage):
            lines.append(f"\n{'='*80}\nStep {step} | AI\n{'='*80}")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                lines.append(f"Content: {content}")
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                lines.append(f"Tool calls ({len(tool_calls)}):")
                for tc in tool_calls:
                    name = tc.get("name", tc.get("function", {}).get("name", "?"))
                    args = tc.get("args", {})
                    tc_id = tc.get("id", "")
                    lines.append(f"  -> {name}({args})  [id={tc_id}]")

        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "unknown")
            tc_id = getattr(msg, "tool_call_id", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"\n{'-'*80}\nStep {step} | TOOL RESULT: {tool_name}  [id={tc_id}]\n{'-'*80}\n{content}")

        else:
            lines.append(f"\nStep {step} | {type(msg).__name__}: {str(msg)}")

    trace = "\n".join(lines)
    if log:
        logger.info(f"\n{'#'*80}\n# AGENT TRACE\n{'#'*80}{trace}\n{'#'*80}\n# END TRACE\n{'#'*80}")
    return trace


def serialize_agent_message(msg: Any, index: int) -> Dict[str, Any]:
    """Serialize a LangChain message without dropping provider/tool metadata."""
    serialized: Dict[str, Any] = {
        "index": index,
        "type": type(msg).__name__,
        "role": getattr(msg, "type", type(msg).__name__.replace("Message", "").lower()),
    }

    for attr in (
        "id", "name", "content", "additional_kwargs", "response_metadata",
        "usage_metadata", "tool_calls", "invalid_tool_calls", "tool_call_id",
        "status", "artifact",
    ):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if attr == "content" or _has_value(value):
                serialized[attr] = json_safe(value)

    return serialized


def extract_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten normalized and provider-native tool calls from serialized messages."""
    tool_calls: List[Dict[str, Any]] = []

    for message in messages:
        normalized = message.get("tool_calls") or []
        for ci, tc in enumerate(normalized):
            call = json_safe(tc)
            if not isinstance(call, dict):
                call = {"value": call}
            call.update({"message_index": message.get("index"), "call_index": ci, "source": "normalized"})
            tool_calls.append(call)

        if normalized:
            continue

        raw_calls = (message.get("additional_kwargs") or {}).get("tool_calls", [])
        for ci, tc in enumerate(raw_calls):
            call = json_safe(tc)
            if not isinstance(call, dict):
                call = {"value": call}
            call.update({"message_index": message.get("index"), "call_index": ci, "source": "provider"})
            tool_calls.append(call)

    return tool_calls


def extract_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten tool result messages from the serialized message stream."""
    results: List[Dict[str, Any]] = []

    for message in messages:
        if message.get("type") != "ToolMessage" and message.get("role") != "tool":
            continue
        entry: Dict[str, Any] = {
            "message_index": message.get("index"),
            "tool": message.get("name", "unknown"),
            "tool_call_id": message.get("tool_call_id"),
            "status": message.get("status"),
            "content": message.get("content", ""),
        }
        if "artifact" in message:
            entry["artifact"] = message["artifact"]
        results.append(entry)

    return results


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, set, dict)):
        return bool(value)
    return True
