import os
from typing import Optional, List
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sparql_llm.utils import SparqlEndpointLinks

# Configuration Class
class Settings(BaseModel):
    # Agent & API Configuration
    mcp_url: str = Field(default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:80/mcp/sse"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower()) # "openai" or "mistral" or "ollama"
    
    # OpenAI Settings
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_endpoint: str = Field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT"))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1"))

    # Mistral Settings
    mistral_api_key: str = Field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))
    mistral_model: str = Field(default_factory=lambda: os.getenv("MISTRAL_MODEL", "ministral-14b-2512"))
    mistral_endpoint: Optional[str] = Field(default_factory=lambda: os.getenv("MISTRAL_ENDPOINT", None))
    
    # Vector DB & Embeddings
    vector_store_type: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "memory_embedding")) # "qdrant" or "memory" or "memory_embedding"
    docs_collection_name: str = Field(default_factory=lambda: os.getenv("DOCS_COLLECTION_NAME", "sparql_endpoint_docs"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "embeddinggemma")) # change as needed to any model supported by the provider
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "ollama")) # "fastembed" or "ollama"
    embedding_dimensions: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768"))) # e.g., 768 for embeddinggemma, adjust as needed
    
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral-nemo"))

    qdrant_host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    
    default_number_of_retrieved_docs: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_RETRIEVED_DOCS", "3")))
    force_index: bool = Field(default_factory=lambda: os.getenv("FORCE_INDEX", "false").lower() == "true")
    auto_init: bool = Field(default_factory=lambda: os.getenv("AUTO_INIT", "true").lower() == "true")
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))

    # Legacy Tools Settings
    enable_legacy_tools: bool = Field(default_factory=lambda: os.getenv("ENABLE_LEGACY_TOOLS", "false").lower() == "true")
    nominatim_endpoint: str = Field(default_factory=lambda: os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org/"))
    
    # Agent Tools Configuration
    enabled_tools: Optional[List[str]] = Field(default_factory=lambda: [t.strip() for t in os.getenv("ENABLED_TOOLS").split(",")] if os.getenv("ENABLED_TOOLS") else None)

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
