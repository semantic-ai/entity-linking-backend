from qdrant_client.models import ScoredPoint
from helpers import logger
from src.config import settings
from src.agent import Agent
from src.agent_helpers.models import AgentConfig

def format_docs(docs: list[ScoredPoint]) -> str:
    """Format a list of documents."""
    return "\n".join(_format_doc(doc) for doc in docs)


def _format_doc(doc: ScoredPoint) -> str:
    """Format a single document, with special formatting based on doc type (sparql, schema)."""
    if not doc.payload:
        return ""
    doc_meta: dict[str, str] = doc.payload.get("metadata", {})
    if doc_meta.get("answer"):
        doc_lang = ""
        doc_type = str(doc_meta.get("doc_type", "")).lower()
        if "query" in doc_type:
            doc_lang = f"sparql\n#+ endpoint: {doc_meta.get('endpoint_url', 'undefined')}"
        elif "schema" in doc_type:
            doc_lang = "shex"
        return f"{doc.payload['page_content']}:\n\n```{doc_lang}\n{doc_meta.get('answer')}\n```"
    # Generic formatting:
    meta = "".join(f" {k}={v!r}" for k, v in doc_meta.items())
    if meta:
        meta = f" {meta}"
    return f"{meta}\n{doc.payload['page_content']}\n"

def initialize_agent() -> Agent:
    # Initialize Agent
    api_key, endpoint, model = settings.get_llm_config()
    logger.info(f"Initializing Agent with provider={settings.llm_provider}, model={model}, endpoint={endpoint}")
    agent_conf = AgentConfig(
        mcp_server_url=settings.mcp_url,
        provider=settings.llm_provider,
        api_key=api_key,
        endpoint=endpoint, # Can be None for Mistral
        model=model,
        temperature=settings.temperature,
        verbose=settings.verbose,
        agent_enabled_tools=settings.agent_enabled_tools,
        llm_max_retries=settings.llm_max_retries,
        llm_request_timeout=settings.llm_request_timeout,
        research_timeout=settings.research_timeout,
        retrieve_tools=settings.research_retrieve_tools,
        execute_tools=settings.research_execute_tools,
        step_timeout_s=settings.research_step_timeout_s,
        max_step_tool_calls=settings.research_max_step_tool_calls,
        max_replans=settings.research_max_replans,
    )
    agent_instance = Agent(agent_conf)
    logger.info("Agent initialized successfully")
    return agent_instance
