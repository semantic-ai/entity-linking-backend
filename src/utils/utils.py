
import time
from qdrant_client.models import ScoredPoint
from helpers import logger, query
from config.config import settings
from src.agent import Agent, AgentConfig

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
        verbose=True,
        enabled_tools=settings.enabled_tools
    )
    agent_instance = Agent(agent_conf)
    logger.info("Agent initialized successfully")
    return agent_instance

# ==============================================================================
# SPARQL NAMESPACE PREFIXES
# ==============================================================================
# Maps prefix names to their full URIs for use in SPARQL queries

SPARQL_PREFIXES = {
    "mu": "http://mu.semte.ch/vocabularies/core/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "airo": "https://w3id.org/airo#",
    "example": "http://www.example.org/",
    "ex": "http://example.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "lblod": "https://data.vlaanderen.be/ns/lblod#",
    "oa": "http://www.w3.org/ns/oa#",
    "dct": "http://purl.org/dc/terms/",
    "dcterms": "http://purl.org/dc/terms/",
    "skolem": "http://www.example.org/id/.well-known/genid/",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "locn": "http://www.w3.org/ns/locn#",
    "geosparql": "http://www.opengis.net/ont/geosparql#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "adms": "http://www.w3.org/ns/adms#",
    "task": "http://redpencil.data.gift/vocabularies/tasks/",
    "nfo": "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#",
    "eli": "http://data.europa.eu/eli/ontology#",
    "ns1": "http://www.w3.org/ns/dqv#",
    "ns2": "https://w3id.org/okn/o/sd#",
    "ns3": "https://w3id.org/airo#",
    "schema": "https://schema.org/",
    "epvoc": "https://data.europarl.europa.eu/def/epvoc#",
    "nie": "http://www.semanticdesktop.org/ontologies/2007/01/19/nie#",
    "harvesting": "http://lblod.data.gift/vocabularies/harvesting/"
}

def get_prefixes_for_query(*prefix_names: str) -> str:
    """
    Generate a SPARQL PREFIX section for only the specified prefixes.

    Args:
        *prefix_names: Variable number of prefix names to include

    Returns:
        A string containing the requested PREFIX declarations

    Example:
        >>> query = get_prefixes_for_query("oa", "prov", "mu")
        >>> query += "SELECT ?s WHERE { ... }"
    """
    lines = []
    for prefix_name in prefix_names:
        if prefix_name in SPARQL_PREFIXES:
            uri = SPARQL_PREFIXES[prefix_name]
            lines.append("PREFIX {0}: <{1}>".format(prefix_name, uri))
    if not lines:
        raise ValueError(f"No valid prefixes found in: {prefix_names}")
    return "\n".join(lines) + "\n"


def wait_for_triplestore():
    triplestore_live = False
    logger.info("Waiting for triplestore...")
    while not triplestore_live:
        try:
            result = query(
                """
                SELECT ?s WHERE {
                ?s ?p ?o.
                } LIMIT 1""",
            )
            if result["results"]["bindings"][0]["s"]["value"]:
                triplestore_live = True
            else:
                raise Exception("triplestore not ready yet...")
        except Exception as _e:
            logger.info(f"Triplestore not live yet, retrying...")
            time.sleep(1)
    logger.info("Triplestore ready!")