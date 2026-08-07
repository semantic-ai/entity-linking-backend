"""JSON serialization helpers for graph responses and SSE events."""

from typing import Any

from pydantic import BaseModel


def json_safe(value: Any) -> Any:
    """Convert LangChain and Pydantic values to JSON-safe structures."""
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
