import os
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from qdrant_client import QdrantClient
from sparql_llm.utils import SparqlEndpointLinks

NEL_TASK_OPERATION = "http://lblod.data.gift/id/jobs/concept/TaskOperation/named-entity-linking"

CONFIG_FILE = Path(os.getenv("CONFIG_FILE", "/config/config.json"))


class Settings(BaseModel):
    """Service configuration. Each field reads its default from an env var with a literal fallback."""



    # Logging and tracing
    verbose: bool = os.getenv("VERBOSE", "false").lower() == "true"
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() == "true"

    # Agent & API
    mcp_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:80/mcp/sse")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    streaming_enabled: bool = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_endpoint: Optional[str] = os.getenv("OPENAI_ENDPOINT")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

    # Mistral
    mistral_api_key: Optional[str] = os.getenv("MISTRAL_API_KEY")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "ministral-14b-2512")
    mistral_endpoint: Optional[str] = os.getenv("MISTRAL_ENDPOINT")

    # Vector DB & embeddings
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "memory_embedding")
    docs_collection_name: str = os.getenv("DOCS_COLLECTION_NAME", "sparql_endpoint_docs")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

    # Elasticsearch vector search
    search_endpoint: str = os.getenv("SEARCH_ENDPOINT", "http://search")
    embedding_api_url: Optional[str] = os.getenv("EMBEDDING_API_URL")
    elastic_search_k: int = int(os.getenv("ELASTIC_SEARCH_K", "30"))
    elastic_search_num_candidates: int = int(os.getenv("ELASTIC_SEARCH_NUM_CANDIDATES", "100"))
    elastic_search_min_score: float = float(os.getenv("ELASTIC_SEARCH_MIN_SCORE", "0.72"))
    elastic_search_keyword_fields: str = os.getenv("ELASTIC_SEARCH_KEYWORD_FIELDS", "description.*")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "10.0"))
    title_fallback_chars: int = int(os.getenv("TITLE_FALLBACK_CHARS", "80"))

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral-nemo")

    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    default_number_of_retrieved_docs: int = int(os.getenv("DEFAULT_RETRIEVED_DOCS", "3"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    force_index: bool = os.getenv("FORCE_INDEX", "false").lower() == "true"
    auto_init: bool = os.getenv("AUTO_INIT", "true").lower() == "true"
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


    llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    research_timeout: int = int(os.getenv("RESEARCH_TIMEOUT", "600"))
    research_recursion_limit: int = int(os.getenv("RESEARCH_RECURSION_LIMIT", "30"))

    # Legacy tools
    enable_legacy_tools: bool = os.getenv("ENABLE_LEGACY_TOOLS", "false").lower() == "true"
    nominatim_endpoint: str = os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org/")

    # Tool selection
    enabled_tools: List[str] = [
        t.strip()
        for t in os.getenv(
            "ENABLED_TOOLS",
            "search_location,search_sparql_docs,execute_sparql_query,search_expressions",
        ).split(",")
        if t.strip()
    ]

    # Stack
    mu_sparql_endpoint: str = os.getenv("MU_SPARQL_ENDPOINT", "http://virtuoso:8890/sparql")

    def get_llm_config(self):
        """Return (api_key, endpoint, model) for the configured LLM provider."""
        if self.llm_provider == "mistral":
            return self.mistral_api_key, self.mistral_endpoint, self.mistral_model
        elif self.llm_provider == "ollama":
            return None, self.ollama_host, self.ollama_model
        return self.openai_api_key, self.openai_endpoint, self.openai_model


settings = Settings()


# Nested structures (endpoints, entity_class_configs) come from the JSON config file.
_file_config: dict = {}
if CONFIG_FILE.exists():
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as _f:
            _file_config = json.load(_f)
    except Exception as e:
        print(f"Error loading config from {CONFIG_FILE}: {e}")


endpoints: List[SparqlEndpointLinks] = []
if isinstance(_file_config.get("endpoints"), list):
    for ep_config in _file_config["endpoints"]:
        try:
            endpoints.append(SparqlEndpointLinks(**ep_config))
        except Exception as e:
            print(f"Error loading endpoint config: {e}")
if not endpoints:
    endpoints = [
        SparqlEndpointLinks(
            endpoint_url=settings.mu_sparql_endpoint,
            void_file="data/queries/local/local_sparql_void.ttl",
            examples_file="data/queries/local/local_sparql_examples.ttl",
            shapes_folder="data/queries/shacl",
        )
    ]

qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
