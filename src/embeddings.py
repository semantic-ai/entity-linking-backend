from typing import List, Iterable
from fastembed import TextEmbedding
import logging

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str, provider: str = "fastembed", base_url: str = None, **kwargs):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "fastembed":
            # Using fastembed directly
            self.model = TextEmbedding(model_name=model_name, **kwargs)
        elif provider == "ollama":
            if not OllamaEmbeddings:
                raise ImportError("langchain-ollama is required for Ollama embeddings")
            # Using generic Ollama wrapper
            self.model = OllamaEmbeddings(model=model_name, base_url=base_url, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed(self, texts: List[str]) -> Iterable[List[float]]:
        if self.provider == "fastembed":
            return self.model.embed(texts)
        elif self.provider == "ollama":
            # OllamaEmbeddings.embed_documents returns List[List[float]]
            return self.model.embed_documents(texts)
        return []
