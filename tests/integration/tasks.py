"""Seed tasks and run them the way the service does."""
import pathlib
import re
from collections import defaultdict

from decide_ai_service_base.sparql_config import GRAPHS, JOB_STATUSES
from decide_ai_service_base.task import Task
from decide_ai_service_base.util import write_agent_info
from helpers import query, update
from rdflib import Graph

import src.task  # noqa: F401  registers NamedEntityLinkingTask

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SUCCESS = JOB_STATUSES["success"]
FAILED = JOB_STATUSES["failed"]

SERVICE_BASE = "http://lblod.data.gift/id/components/named-entity-linking/v1.0.0"

# The graph the service's queries read each fixture subject from.
GRAPH_BY_SUBJECT_PREFIX = {
    "http://example.org/test/task/": "jobs",
    "http://example.org/test/job/": "jobs",
    "http://example.org/test/container/": "jobs",
    "http://example.org/test/governing-unit/": "public",
    "http://example.org/test/annotation/": "ai",
    "http://example.org/test/statement/": "ai",
    "http://example.org/test/target/": "ai",
    "http://example.org/test/entity/": "ai",
}

INSERT_INTO_GRAPH = "INSERT DATA { GRAPH <%s> { %s } }"

PREFIXES = """
PREFIX oa: <http://www.w3.org/ns/oa#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX locn: <http://www.w3.org/ns/locn#>
PREFIX geosparql: <http://www.opengis.net/ont/geosparql#>
"""

TASK_STATUS = """
SELECT ?status WHERE {
  GRAPH <%s> { <%s> <http://www.w3.org/ns/adms#status> ?status }
}
"""

LINKS = PREFIXES + """
SELECT ?link WHERE {
  GRAPH <%s> {
    ?annotation oa:hasTarget <%s> ;
                oa:hasBody ?statement .
    ?statement rdf:predicate skos:exactMatch ;
               rdf:object ?link .
  }
}
"""

SEARCH_SCORES = PREFIXES + """
SELECT ?score WHERE {
  GRAPH <%s> {
    ?annotation oa:hasTarget <%s> ;
                oa:hasBody ?statement ;
                <http://mu.semte.ch/vocabularies/ext/muSearchScore> ?score .
    ?statement rdf:predicate skos:exactMatch .
  }
}
"""

ENTITY = "SELECT ?p ?o WHERE { GRAPH <%s> { <%s> ?p ?o } }"

WKT = PREFIXES + """
SELECT ?wkt WHERE {
  GRAPH <%s> { <%s> locn:geometry / geosparql:asWKT ?wkt }
}
"""


def bindings(sparql: str) -> list[dict]:
    return query(sparql, sudo=True)["results"]["bindings"]


def _graph_for(subject: str) -> str:
    for prefix, key in GRAPH_BY_SUBJECT_PREFIX.items():
        if subject.startswith(prefix):
            return key
    raise ValueError(f"no graph for fixture subject {subject}")


def seed(fixture_name: str) -> None:
    buckets = defaultdict(Graph)
    for triple in Graph().parse(FIXTURES / f"{fixture_name}.ttl"):
        buckets[_graph_for(str(triple[0]))].add(triple)

    for key, data in buckets.items():
        update(INSERT_INTO_GRAPH % (GRAPHS[key], data.serialize(format="nt")), sudo=True)


def run(task_uri: str) -> None:
    # Same registration as web.py startup.
    write_agent_info(SERVICE_BASE)
    Task.from_uri(task_uri).execute()


def status(task_uri: str) -> str:
    rows = bindings(TASK_STATUS % (GRAPHS["jobs"], task_uri))
    assert len(rows) == 1, f"expected one status for {task_uri}, got {len(rows)}"
    return rows[0]["status"]["value"]


def links(target_uri: str) -> list[str]:
    """skos:exactMatch objects of the linking annotations on ``target_uri``, one per annotation."""
    return [row["link"]["value"] for row in bindings(LINKS % (GRAPHS["ai"], target_uri))]


def search_scores(target_uri: str) -> list[float]:
    return [float(row["score"]["value"]) for row in bindings(SEARCH_SCORES % (GRAPHS["ai"], target_uri))]


def entity(entity_uri: str) -> set[tuple[str, str]]:
    return {(row["p"]["value"], row["o"]["value"]) for row in bindings(ENTITY % (GRAPHS["ai"], entity_uri))}


def wkt(entity_uri: str) -> str:
    [row] = bindings(WKT % (GRAPHS["ai"], entity_uri))
    return row["wkt"]["value"]


def coordinates(entity_uri: str) -> set[float]:
    # Virtuoso rewrites the literal and swaps the axes of EPSG:4326 coordinates, so only the values compare.
    return {float(n) for n in re.findall(r"-?\d+\.\d+", wkt(entity_uri))}
