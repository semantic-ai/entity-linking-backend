"""Elasticsearch hybrid search tool.

Two-step pipeline:
  1. ``search_expressions`` – runs kNN, BM25, or both against
     mu-search's ``/expressions/search`` endpoint.
  2. ``fetch_documents`` – enriches the raw hits (uri, score) with title,
     content, and download URL from Virtuoso via SPARQL.
"""

import logging
from typing import Optional

import httpx

from src.config import settings
from src.tools.sparql_search import SparqlClient

logger = logging.getLogger(__name__)

# Enrichment query: fetches title, content, and download URL for a set of
# eli:Expression URIs. ``{values}`` is replaced at call time with
# space-separated ``<uri>`` tokens.
_ENRICHMENT_QUERY = """\
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX epvoc: <https://data.europarl.europa.eu/def/epvoc#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX oa: <http://www.w3.org/ns/oa#>
SELECT ?s ?title ?content ?downloadUrl
WHERE {
  VALUES ?s { {values} }
  ?s a eli:Expression .
  { ?s eli:title ?title . }
  UNION
  { ?annotation oa:hasTarget / oa:hasSource ?s .
    ?annotation oa:hasBody ?body .
    ?body rdf:predicate eli:title .
    ?body rdf:object ?title . }
  UNION
  { ?s a eli:Expression . }
  { ?s <http://www.w3.org/ns/prov#wasDerivedFrom> ?downloadUrl . }
  UNION
  { ?s <http://data.europa.eu/eli/ontology#is_embodied_by> / <http://data.europa.eu/eli/ontology#is_exemplified_by> ?downloadUrl .
    FILTER(!STRSTARTS(STR(?downloadUrl), "http://internal-files/")) }
  UNION
  { ?s a eli:Expression . }
  OPTIONAL { ?s epvoc:expressionContent ?content . }
}"""



async def get_embedding(text: str) -> list[float]:
    if not settings.embedding_api_url:
        raise ValueError("EMBEDDING_API_URL is required for vector expression search.")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.embedding_api_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
            headers={"Content-Type": "application/json"},
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [[]])
        return embeddings[0] if embeddings else []
    

async def search_expressions(
    question: str,
    top_n: int = 5,
    local_authority: Optional[str] = None,
    keyword: Optional[str] = None,
    vector: bool = True,
) -> list[dict]:
    """Search the expressions index using kNN, BM25, or both.

    When *vector* is ``True``, embeds *question* and runs a kNN query.
    When *keyword* is provided, adds a ``multi_match`` BM25 clause.
    When *vector* is ``False`` and *keyword* is given, uses pure BM25.
    At least one of *vector* / *keyword* must be active.
    Results below ``settings.elastic_search_min_score`` are discarded.

    Args:
        question: The natural-language question to search for.
        top_n: Maximum number of results to return.
        local_authority: URI of the owning body to filter by (optional).
        keyword: Optional keyword string for BM25 ``multi_match`` (optional).
        vector: Whether to include the kNN vector search (default ``True``).

    Returns:
        A list of dicts with ``uri`` and ``score`` keys, capped at *top_n* items.
    """
    if not vector and not keyword:
        raise ValueError("At least one of 'vector' or 'keyword' must be truthy.")
    if not 1 <= top_n <= 20:
        raise ValueError("top_n must be between 1 and 20.")

    keyword_fields = [f.strip() for f in settings.elastic_search_keyword_fields.split(",") if f.strip()]
    if keyword and not keyword_fields:
        raise ValueError("ELASTIC_SEARCH_KEYWORD_FIELDS must define at least one field.")

    bool_query: dict = {}

    if vector:
        embedding = await get_embedding(question)
        knn_clause = {
            "knn": {
                "field": "description-vector",
                "query_vector": embedding,
                "k": settings.elastic_search_k,
                "num_candidates": settings.elastic_search_num_candidates,
            }
        }
        bool_query["must"] = [knn_clause]
        if keyword:
            bool_query["should"] = [{"multi_match": {"query": keyword, "fields": keyword_fields, "fuzziness": "AUTO"}}]
    else:
        bool_query["must"] = [{"multi_match": {"query": keyword, "fields": keyword_fields, "fuzziness": "AUTO"}}]

    if local_authority:
        bool_query["filter"] = [{"term": {"owning-body": local_authority}}]

    search_url = f"{settings.search_endpoint.rstrip('/')}/expressions/search"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            search_url,
            json={"query": {"bool": bool_query}, "size": top_n},
            headers={"Content-Type": "application/json"},
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        data = response.json().get("data", [])

    results = [
        {"uri": doc["attributes"]["uri"], "score": doc.get("score")}
        for doc in data
        if doc.get("attributes", {}).get("uri")
    ]
    results = [r for r in results if r["score"] is None or r["score"] >= settings.elastic_search_min_score]
    return await fetch_documents(results[:top_n])


async def fetch_documents(results: list[dict]) -> list[dict]:
    """Enrich search results with document metadata from Virtuoso.

    Queries the configured SPARQL endpoint for title, content, and download URL
    for each URI in *results*. Falls back to a content snippet as title when no
    explicit title is available.

    Args:
        results: Dicts with at least ``uri`` and ``score`` keys.

    Returns:
        The input list extended with ``title``, ``content``, and
        ``download_url`` fields (``None`` when unavailable).
    """
    if not results:
        return []

    values = " ".join(f"<{r['uri']}>" for r in results)
    sparql_query = _ENRICHMENT_QUERY.replace("{values}", values)

    sparql_client = SparqlClient(endpoint=settings.mu_sparql_endpoint)
    bindings = await sparql_client.search(query=sparql_query)

    doc_map: dict[str, dict] = {}
    for binding in bindings:
        uri = binding.get("s", {}).get("value")
        if not uri:
            continue
        entry = doc_map.setdefault(uri, {"title": None, "content": None, "download_url": None})
        title = binding.get("title", {}).get("value")
        content = binding.get("content", {}).get("value")
        download_url = binding.get("downloadUrl", {}).get("value")
        if content and not entry["content"]:
            entry["content"] = " ".join(content.split())
        if title and title.strip() and not (entry["title"] and entry["title"].strip()):
            entry["title"] = title.strip()
        if download_url and download_url.strip() and not entry["download_url"]:
            entry["download_url"] = download_url.strip()

    # Fall back to a content snippet when no explicit title was found.
    for entry in doc_map.values():
        if not (entry["title"] and entry["title"].strip()) and entry["content"]:
            snippet = " ".join(entry["content"].split()).lstrip("*# ").strip()
            entry["title"] = snippet[:settings.title_fallback_chars] or None

    _empty = {"title": None, "content": None, "download_url": None}
    return [{"uri": r["uri"], "score": r["score"], **doc_map.get(r["uri"], _empty)} for r in results]
