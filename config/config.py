import os
import json
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sparql_llm.utils import SparqlEndpointLinks

# Load external configuration
CONFIG_FILE = os.getenv("CONFIG_FILE", "/config/config.json")
file_config = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            file_config = json.load(f)
            print(f"Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        print(f"Error loading config from {CONFIG_FILE}: {e}")

def get_config_value(env_key: str, config_key: str, default: Any) -> Any:
    """
    Retrieve setting with priority: Environment Variable > Config File > Default.
    """
    val = os.getenv(env_key)
    if val is not None:
        return val
    if config_key in file_config:
        return file_config[config_key]
    return default

def get_config_bool(env_key: str, config_key: str, default: str) -> bool:
    val = get_config_value(env_key, config_key, default)
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"

def get_config_int(env_key: str, config_key: str, default: str) -> int:
    val = get_config_value(env_key, config_key, default)
    return int(val)

def get_config_float(env_key: str, config_key: str, default: str) -> float:
    val = get_config_value(env_key, config_key, default)
    return float(val)

def get_config_list(env_key: str, config_key: str, default: Optional[List[str]] = None) -> Optional[List[str]]:
    val = os.getenv(env_key)
    if val:
        return [t.strip() for t in val.split(",")]
    if config_key in file_config and isinstance(file_config[config_key], list):
        return file_config[config_key]
    if config_key in file_config and isinstance(file_config[config_key], str):
        return [t.strip() for t in file_config[config_key].split(",")]
    return default

# Configuration Class
# Settings are resolved with the following priority (highest to lowest):
# 1. Environment Variables (e.g., set in .env or Docker environment)
# 2. Config File (values loaded from the external JSON configuration file)
# 3. Default Values (defaults provided in the Field definitions below)
class Settings(BaseModel):
    # Agent & API Configuration
    mcp_url: str = Field(default_factory=lambda: get_config_value("MCP_SERVER_URL", "mcp_url", "http://localhost:80/mcp/sse"))
    llm_provider: str = Field(default_factory=lambda: get_config_value("LLM_PROVIDER", "llm_provider", "openai").lower()) # "openai" or "mistral" or "ollama"
    
    # OpenAI Settings
    openai_api_key: str = Field(default_factory=lambda: get_config_value("OPENAI_API_KEY", "openai_api_key", None))
    openai_endpoint: str = Field(default_factory=lambda: get_config_value("OPENAI_ENDPOINT", "openai_endpoint", None))
    openai_model: str = Field(default_factory=lambda: get_config_value("OPENAI_MODEL", "openai_model", "gpt-4.1"))

    # Mistral Settings
    mistral_api_key: str = Field(default_factory=lambda: get_config_value("MISTRAL_API_KEY", "mistral_api_key", None))
    mistral_model: str = Field(default_factory=lambda: get_config_value("MISTRAL_MODEL", "mistral_model", "ministral-14b-2512"))
    mistral_endpoint: Optional[str] = Field(default_factory=lambda: get_config_value("MISTRAL_ENDPOINT", "mistral_endpoint", None))
    
    # Vector DB & Embeddings
    vector_store_type: str = Field(default_factory=lambda: get_config_value("VECTOR_STORE_TYPE", "vector_store_type", "memory_embedding")) # "qdrant" or "memory" or "memory_embedding"
    docs_collection_name: str = Field(default_factory=lambda: get_config_value("DOCS_COLLECTION_NAME", "docs_collection_name", "sparql_endpoint_docs"))
    embedding_model: str = Field(default_factory=lambda: get_config_value("EMBEDDING_MODEL", "embedding_model", "embeddinggemma")) # change as needed to any model supported by the provider
    embedding_provider: str = Field(default_factory=lambda: get_config_value("EMBEDDING_PROVIDER", "embedding_provider", "ollama")) # "fastembed" or "ollama"
    embedding_dimensions: int = Field(default_factory=lambda: get_config_int("EMBEDDING_DIMENSIONS", "embedding_dimensions", "768")) # e.g., 768 for embeddinggemma, adjust as needed
    
    ollama_host: str = Field(default_factory=lambda: get_config_value("OLLAMA_HOST", "ollama_host", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: get_config_value("OLLAMA_MODEL", "ollama_model", "mistral-nemo"))

    qdrant_host: str = Field(default_factory=lambda: get_config_value("QDRANT_HOST", "qdrant_host", "localhost"))
    qdrant_port: int = Field(default_factory=lambda: get_config_int("QDRANT_PORT", "qdrant_port", "6333"))
    
    default_number_of_retrieved_docs: int = Field(default_factory=lambda: get_config_int("DEFAULT_RETRIEVED_DOCS", "default_number_of_retrieved_docs", "3"))
    force_index: bool = Field(default_factory=lambda: get_config_bool("FORCE_INDEX", "force_index", "false"))
    auto_init: bool = Field(default_factory=lambda: get_config_bool("AUTO_INIT", "auto_init", "true"))
    temperature: float = Field(default_factory=lambda: get_config_float("TEMPERATURE", "temperature", "0.0"))

    # Legacy Tools Settings
    enable_legacy_tools: bool = Field(default_factory=lambda: get_config_bool("ENABLE_LEGACY_TOOLS", "enable_legacy_tools", "false"))
    nominatim_endpoint: str = Field(default_factory=lambda: get_config_value("NOMINATIM_ENDPOINT", "nominatim_endpoint", "https://nominatim.openstreetmap.org/"))
    
    # Agent Tools Configuration
    enabled_tools: Optional[List[str]] = Field(default_factory=lambda: get_config_list("ENABLED_TOOLS", "enabled_tools", None))

    def get_llm_config(self):
        """Returns the API Key, Endpoint, and Model based on the selected provider."""
        if self.llm_provider == "mistral":
            return self.mistral_api_key, self.mistral_endpoint, self.mistral_model
        elif self.llm_provider == "ollama":
            return None, self.ollama_host, self.ollama_model
        else:
            return self.openai_api_key, self.openai_endpoint, self.openai_model

settings = Settings()

# Endpoints configuration
endpoints = []

# Load endpoints from config file if available
if "endpoints" in file_config and isinstance(file_config["endpoints"], list):
    for ep_config in file_config["endpoints"]:
        try:
             endpoints.append(SparqlEndpointLinks(**ep_config))
        except Exception as e:
             print(f"Error loading endpoint config: {e}")

# Default endpoints if none loaded
if not endpoints:
    endpoints = [
        # SparqlEndpointLinks(
        #     endpoint_url="https://query.wikidata.org/sparql",
        #     void_file="data/queries/wikidata/wikidata_sparql_void.ttl",
        #     examples_file="data/queries/wikidata/wikidata_sparql_examples.ttl",
        # ),
        SparqlEndpointLinks(
            endpoint_url="https://centrale-vindplaats.lblod.info/sparql",
            void_file="data/queries/centrale_vindplaats/centrale_vindplaats_sparql_void.ttl",
            examples_file="data/queries/centrale_vindplaats/centrale_vindplaats_sparql_examples.ttl",
        )
    ]

qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
