"""LangChain callback handler for logging agent progress."""

from langchain_core.callbacks import BaseCallbackHandler
from helpers import logger


class AgentStepLogger(BaseCallbackHandler):
    """Logs each LLM call and tool invocation for visibility into agent progress."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "unknown")
        logger.info(f"[RESEARCH][LLM] Calling model ({model_name})...")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "unknown")
        msg_count = sum(len(batch) for batch in messages) if messages else 0
        logger.info(f"[RESEARCH][LLM] Calling chat model ({model_name}) with {msg_count} messages...")

    def on_llm_end(self, response, **kwargs):
        logger.info("[RESEARCH][LLM] Model responded.")

    def on_llm_error(self, error, **kwargs):
        logger.error(f"[RESEARCH][LLM] Model error: {error}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logger.info(f"[RESEARCH][TOOL] Invoking tool: {tool_name}")

    def on_tool_end(self, output, **kwargs):
        output_preview = str(output)[:200] if output else "(empty)"
        logger.info(f"[RESEARCH][TOOL] Tool returned: {output_preview}...")

    def on_tool_error(self, error, **kwargs):
        logger.error(f"[RESEARCH][TOOL] Tool error: {error}")
