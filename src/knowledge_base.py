from abc import ABC, abstractmethod
import math
import time
import logging
from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from sparql_llm.loaders.sparql_examples_loader import SparqlExamplesLoader
from sparql_llm.loaders.sparql_void_shapes_loader import SparqlVoidShapesLoader
from sparql_llm.utils import get_prefixes_and_schema_for_endpoints
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import settings, qdrant_client, endpoints
from src.embeddings import EmbeddingModel

from helpers import logger


def load_shapes_from_folder(shapes_folder: str, endpoint_url: str) -> List[Document]:
    """Load SHACL shape definitions from .ttl files in a folder (non-recursive)."""
    folder = Path(shapes_folder)
    if not folder.is_dir():
        logger.warning(f"Shapes folder not found: {shapes_folder}")
        return []
    docs = []
    for file in sorted(folder.glob("*.ttl")):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=f"SHACL shape definition from {file.name}",
            metadata={
                "answer": content,
                "doc_type": "SHACL shapes schema",
                "endpoint_url": endpoint_url,
            },
        ))
    logger.info(f"Loaded {len(docs)} SHACL shapes from {shapes_folder}")
    return docs

class KnowledgeBase(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the knowledge base with data."""
        pass

    @abstractmethod
    def search(self, question: str, potential_classes: List[str], steps: List[str]) -> List[Any]:
        """Search the knowledge base for relevant documents."""
        pass


class QdrantKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self.embedding_model = EmbeddingModel(
            settings.embedding_model, 
            provider=settings.embedding_provider,
            base_url=settings.ollama_host
        )
        self.client = qdrant_client

    def initialize(self) -> None:
        """Initialize the vectordb with example queries and ontology descriptions from the SPARQL endpoints"""
        
        # Check if initialization is needed
        collection_needs_init = (
            settings.force_index
            or not self.client.collection_exists(settings.docs_collection_name)
            or not self.client.get_collection(settings.docs_collection_name).points_count
        )
        
        if not collection_needs_init:
            logger.info(
                f"[Qdrant] Collection '{settings.docs_collection_name}' exists with {self.client.get_collection(settings.docs_collection_name).points_count} points. Skipping initialization."
            )
            return

        if not settings.auto_init and collection_needs_init:
            logger.warning(
                f"[Qdrant] Collection '{settings.docs_collection_name}' does not exist or is empty. Run init manually."
            )
            return

        logger.info("[Qdrant] Initializing Qdrant knowledge base...")

        docs: List[Document] = []
        prefix_map, _void_schema = get_prefixes_and_schema_for_endpoints(endpoints)

        # Gets documents from the SPARQL endpoints
        for endpoint in endpoints:
            if endpoint.get("examples_file"):
                 docs += SparqlExamplesLoader(
                    endpoint.get("endpoint_url"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("void_file"):
                docs += SparqlVoidShapesLoader(
                    endpoint.get("endpoint_url"),
                    prefix_map=prefix_map,
                    void_file=endpoint.get("void_file"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("shapes_folder"):
                docs += load_shapes_from_folder(
                    endpoint.get("shapes_folder"),
                    endpoint.get("endpoint_url"),
                )

            logger.info(f"[Qdrant] Generating embeddings for {len(docs)} documents from endpoint {endpoint.get('endpoint_url')}...")
        start_time = time.time()
        
        # Re-create collection
        if self.client.collection_exists(settings.docs_collection_name):
            self.client.delete_collection(settings.docs_collection_name)
            
        self.client.create_collection(
            collection_name=settings.docs_collection_name,
            vectors_config=VectorParams(size=settings.embedding_dimensions, distance=Distance.COSINE),
        )
        
        if not docs:
            logger.info("[Qdrant] No documents found to index.")
            return

        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        
        # Embed
        embeddings = list(self.embedding_model.embed(texts))
        
        points = [
            PointStruct(
                id=i, 
                vector=embedding, 
                payload={"page_content": text, "metadata": meta}
            )
            for i, (text, meta, embedding) in enumerate(zip(texts, metadatas, embeddings))
        ]
        
        # Batch upsert
        if points:
            self.client.upsert(
                collection_name=settings.docs_collection_name,
                points=points
            )
        
        logger.info(f"[Qdrant] Done generating and indexing {len(docs)} documents into the vectordb in {time.time() - start_time} seconds")

    def search(self, question: str, potential_classes: List[str], steps: List[str]) -> List[Any]:
        relevant_docs = []
        
        to_embed = [question] + steps + potential_classes
        search_embeddings = list(self.embedding_model.embed(to_embed))
        
        for search_embedding in search_embeddings:
            # Get SPARQL example queries
            relevant_docs.extend(
                doc
                for doc in self.client.query_points(
                    query=search_embedding,
                    collection_name=settings.docs_collection_name,
                    limit=settings.default_number_of_retrieved_docs,
                    score_threshold=settings.similarity_threshold,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.doc_type",
                                match=MatchValue(value="SPARQL endpoints query examples"),
                            )
                        ]
                    ),
                ).points
                if doc.payload
                and doc.payload.get("metadata", {}).get("answer")
                not in {
                    existing_doc.payload.get("metadata", {}).get("answer") if existing_doc.payload else None
                    for existing_doc in relevant_docs
                }
            )
            # Get other relevant documentation (classes schemas, general information)
            relevant_docs.extend(
                doc
                for doc in self.client.query_points(
                    query=search_embedding,
                    collection_name=settings.docs_collection_name,
                    limit=settings.default_number_of_retrieved_docs,
                    score_threshold=settings.similarity_threshold,
                    query_filter=Filter(
                        must_not=[
                            FieldCondition(
                                key="metadata.doc_type",
                                match=MatchValue(value="SPARQL endpoints query examples"),
                            )
                        ]
                    ),
                ).points
                if doc.payload
                and doc.payload.get("metadata", {}).get("answer")
                not in {
                    existing_doc.payload.get("metadata", {}).get("answer") if existing_doc.payload else None
                    for existing_doc in relevant_docs
                }
            )
        logger.info(f"[Qdrant] Found {len(relevant_docs)} relevant documents for question: '{question}'")
        return relevant_docs

class SimpleKnowledgeBase(KnowledgeBase):
    """Concatenates all shapes and examples into a full context string (notebook-style)."""

    def __init__(self):
        self.context_docs: List[Any] = []

    def initialize(self) -> None:
        logger.info("[SimpleKnowledgeBase] Initializing Simple Knowledge Base...")
        parts = []

        for endpoint in endpoints:
            endpoint_url = endpoint.get("endpoint_url", "")

            # Load SHACL shapes from folder
            if endpoint.get("shapes_folder"):
                folder = Path(endpoint["shapes_folder"])
                if folder.is_dir():
                    for file in sorted(folder.glob("*.ttl")):
                        parts.append(
                            f"\n{'#' * 50}\n"
                            f"# SHACL Shape: {file.name}\n"
                            f"{file.read_text(encoding='utf-8')}\n"
                        )

            # Load examples file(s) raw
            if endpoint.get("examples_file"):
                examples_path = Path(endpoint["examples_file"])
                if examples_path.is_file():
                    parts.append(
                        f"\n{'#' * 50}\n"
                        f"# Examples: {examples_path.name}\n"
                        f"{examples_path.read_text(encoding='utf-8')}\n"
                    )
                elif examples_path.is_dir():
                    for file in sorted(examples_path.glob("*.ttl")):
                        parts.append(
                            f"\n{'#' * 50}\n"
                            f"# Examples: {file.name}\n"
                            f"{file.read_text(encoding='utf-8')}\n"
                        )

        context_string = "\n".join(parts)

        class _ContextDoc:
            def __init__(self, payload):
                self.payload = payload

        if context_string.strip():
            self.context_docs = [_ContextDoc(payload={
                "page_content": "Full SHACL shapes and SPARQL examples context",
                "metadata": {
                    "answer": context_string,
                    "doc_type": "SHACL shapes schema",
                },
            })]

        logger.info(f"[SimpleKnowledgeBase] Loaded {len(parts)} file(s) into context.")

    def search(self, question: str, potential_classes: List[str], steps: List[str]) -> List[Any]:
        logger.info(f"[SimpleKnowledgeBase] Returning {len(self.context_docs)} context document(s) for question: '{question}'")
        return self.context_docs


class LocalKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self.documents: List[Document] = []

    def initialize(self) -> None:
        logger.info("[LocalKnowledgeBase] Initializing Local Knowledge Base (Memory)...")
        self.documents = []
        prefix_map, _void_schema = get_prefixes_and_schema_for_endpoints(endpoints)

        for endpoint in endpoints:
            if endpoint.get("examples_file"):
                self.documents += SparqlExamplesLoader(
                    endpoint.get("endpoint_url"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("void_file"):
                self.documents += SparqlVoidShapesLoader(
                    endpoint.get("endpoint_url"),
                    prefix_map=prefix_map,
                    void_file=endpoint.get("void_file"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("shapes_folder"):
                self.documents += load_shapes_from_folder(
                    endpoint.get("shapes_folder"),
                    endpoint.get("endpoint_url"),
                )
        logger.info(f"[LocalKnowledgeBase] Loaded {len(self.documents)} documents into memory.")

    def search(self, question: str, potential_classes: List[str], steps: List[str]) -> List[Any]:
        results = []
        
        # Combine search terms
        search_terms = set([question] + steps + potential_classes)
        search_terms = {t.lower() for t in search_terms if t}

        for doc in self.documents:

            
            # Simple scoring logic: check if any potential class or step is in the content
            # Or just return all if the user hinted at that.
            # "returns them all or based on potential classes"
            
            # Use a mock class to mimic ScoredPoint
            class MockScoredPoint:
                def __init__(self, payload):
                    self.payload = payload
            
            payload = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            
            # Filtering logic (Naive)
            # If doc matches potential classes in its content or metadata
            add_doc = False
            if not potential_classes:
                 add_doc = True 
            
            # Check if any potential class is in the content
            content_lower = doc.page_content.lower()
            metadata_str = str(doc.metadata).lower()
            
            if any(pc.lower() in content_lower for pc in potential_classes):
                add_doc = True
            elif any(pc.lower() in metadata_str for pc in potential_classes):
                add_doc = True
            
            
            if not potential_classes and not steps:
                add_doc = True
            
            if add_doc:
                results.append(MockScoredPoint(payload=payload))

        # If too many, maybe limit?
        logger.info(f"[LocalKnowledgeBase] Found {len(results)} relevant documents for question: '{question}'")
        return results

class LocalEmbeddingKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self.embedding_model = EmbeddingModel(
            settings.embedding_model, 
            provider=settings.embedding_provider,
            base_url=settings.ollama_host,
        )
        self.documents = []

    def initialize(self) -> None:
        logger.info("[LocalEmbeddingKnowledgeBase] Initializing In-Memory Embedding Knowledge Base...")
        docs: List[Document] = []
        prefix_map, _void_schema = get_prefixes_and_schema_for_endpoints(endpoints)

        for endpoint in endpoints:
            logger.info(f"[LocalEmbeddingKnowledgeBase] Loading documents from endpoint {endpoint.get('endpoint_url')}...")
            if endpoint.get("examples_file"):
                docs += SparqlExamplesLoader(
                    endpoint.get("endpoint_url"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("void_file"):
                docs += SparqlVoidShapesLoader(
                    endpoint.get("endpoint_url"),
                    prefix_map=prefix_map,
                    void_file=endpoint.get("void_file"),
                    examples_file=endpoint.get("examples_file"),
                ).load()

            if endpoint.get("shapes_folder"):
                docs += load_shapes_from_folder(
                    endpoint.get("shapes_folder"),
                    endpoint.get("endpoint_url"),
                )
        logger.info(f"[LocalEmbeddingKnowledgeBase] Loaded {len(docs)} documents into memory.")
        if not docs:
            logger.info("[LocalEmbeddingKnowledgeBase] No documents found to index.")
            return

        
        start_time = time.time()
        
        texts = [d.page_content for d in docs]
        embeddings = list(self.embedding_model.embed(texts))
        
        self.documents = list(zip(docs, embeddings))
        
        logger.info(f"[LocalEmbeddingKnowledgeBase] Done generating embeddings for {len(docs)} documents in {time.time() - start_time} seconds")

    def search(self, question: str, potential_classes: List[str], steps: List[str]) -> List[Any]:
        results = []
        to_embed = [question] + steps + potential_classes
        to_embed = [x for x in to_embed if x]
        
        if not to_embed:
            return []

        search_embeddings = list(self.embedding_model.embed(to_embed))
        seen_answers = set()
        
        def cosine_similarity(v1, v2):
            dot = sum(a*b for a,b in zip(v1, v2))
            mag1 = math.sqrt(sum(a*a for a in v1))
            mag2 = math.sqrt(sum(b*b for b in v2))
            if mag1 == 0 or mag2 == 0: return 0.0
            return dot / (mag1 * mag2)

        class MockScoredPoint:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        for search_emb in search_embeddings:
            scores = []
            for doc, doc_emb in self.documents:
                score = cosine_similarity(search_emb, doc_emb)
                scores.append((score, doc))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # 1. Example queries
            added = 0
            for score, doc in scores:
                if added >= settings.default_number_of_retrieved_docs: break
                if doc.metadata.get("doc_type") == "SPARQL endpoints query examples":
                    ans = doc.metadata.get("answer")
                    if ans not in seen_answers:
                        if ans: seen_answers.add(ans)
                        results.append(MockScoredPoint(
                            payload={"page_content": doc.page_content, "metadata": doc.metadata}, 
                            score=score
                        ))
                        added += 1

            # 2. Others
            added = 0
            for score, doc in scores:
                if added >= settings.default_number_of_retrieved_docs: break
                if doc.metadata.get("doc_type") != "SPARQL endpoints query examples":
                    ans = doc.metadata.get("answer")
                    if ans not in seen_answers:
                        if ans: seen_answers.add(ans)
                        results.append(MockScoredPoint(
                            payload={"page_content": doc.page_content, "metadata": doc.metadata}, 
                            score=score
                        ))
                        added += 1
        logger.info(f"[LocalEmbeddingKnowledgeBase] Found {len(results)} relevant documents for question: '{question}'")
        return results

# Factory to get the KB
def get_knowledge_base() -> KnowledgeBase:
    if settings.vector_store_type == "simple":
        return SimpleKnowledgeBase()
    
    elif settings.vector_store_type == "memory":
        return LocalKnowledgeBase()
    
    elif settings.vector_store_type == "memory_embedding":
        return LocalEmbeddingKnowledgeBase()
    else:
        return QdrantKnowledgeBase()
