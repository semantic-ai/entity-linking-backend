import uuid
from datetime import datetime
from string import Template
from typing import Optional

from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_float
from helpers import update

from decide_ai_service_base.annotation.relation_extraction import RelationExtractionAnnotation
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, SPARQL_PREFIXES


class NamedEntityLinkingAnnotation(RelationExtractionAnnotation):
    """
    NEL annotation (skos:exactMatch linking) that:
    - Reuses the upstream NER annotation's existing oa:hasTarget rather than minting a new
      oa:SpecificResource (resolved from GRAPHS["ai"] via `prev_annotation_uri`).
    - Emits prov:used + prov:startedAtTime + prov:endedAtTime on the activity.
    - Supports extra triples (e.g. Nominatim-derived geospatial enrichment) via get_extra_inserts().
    - Is idempotent: FILTER NOT EXISTS over (subject, predicate, object, target).
    """

    def __init__(
        self,
        prev_annotation_uri: str,
        subject_uri: str,
        object_uri: str,
        activity_id: str,
        source_uri: str,
        agent: str,
        agent_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        extra_triples: str = "",
        confidence: float = 1.0,
    ):
        super().__init__(
            subject=subject_uri,
            predicate="skos:exactMatch",
            obj=sparql_escape_uri(object_uri),
            activity_id=activity_id,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=agent,
            agent_type=agent_type,
            confidence=confidence,
        )
        self.prev_annotation_uri = prev_annotation_uri
        self.start_time = start_time
        self.end_time = end_time
        self.extra_triples = extra_triples

    def get_extra_inserts(self) -> str:
        return self.extra_triples or ""

    def add_to_triplestore_if_not_exists(self) -> str:
        annotation_uuid = str(uuid.uuid4())
        annotation_uri = "{0}{1}".format(SPARQL_PREFIXES["annotations"], annotation_uuid)
        statements_uri = sparql_escape_uri(
            "{0}{1}".format(SPARQL_PREFIXES["statements"], uuid.uuid4())
        )

        statement_parts, statement_filter = self._build_statement_parts(
            statements_uri,
            sparql_escape_uri(self.subject),
            self.predicate,
            self.object,
        )

        activity_id_escaped = sparql_escape_uri(self.activity_id)
        activity_uuid_literal = self.activity_id.rsplit("/", 1)[-1]

        timing_triples = ""
        if self.start_time is not None:
            timing_triples += (
                f"\n                  {activity_id_escaped} prov:startedAtTime "
                f"{sparql_escape_string(self.start_time.isoformat())}^^xsd:dateTime ."
            )
        if self.end_time is not None:
            timing_triples += (
                f"\n                  {activity_id_escaped} prov:endedAtTime "
                f"{sparql_escape_string(self.end_time.isoformat())}^^xsd:dateTime ."
            )

        ai_graph = GRAPHS["ai"]

        query_template = Template(
            get_prefixes_for_query(
                "ext", "oa", "mu", "prov", "rdf", "skos", "xsd",
                "nif", "annotations", "statements",
            )
            + """
            INSERT {
              GRAPH $ai_graph {
                  $activity_id a prov:Activity ;
                     mu:uuid "$activity_uuid" ;
                     prov:generated $annotation_id ;
                     prov:wasAssociatedWith $user ;
                     prov:used ?target .
                  $timing_triples

                  $annotation_id a oa:Annotation ;
                     mu:uuid "$annotation_uuid" ;
                     oa:hasBody $statements_uri ;
                     nif:confidence $confidence ;
                     oa:motivatedBy oa:linking ;
                     oa:hasTarget ?target .

                  $statement_parts

                  $extra
              }
            } WHERE {
              GRAPH $ai_graph {
                  $prev_annotation_uri oa:hasTarget ?target .
              }
              $user <http://www.w3.org/ns/prov#specializationOf> ?modelType .
              FILTER NOT EXISTS {
                GRAPH $ai_graph {
                  ?existingAnn a oa:Annotation ;
                      oa:hasBody ?existingStatement ;
                      oa:motivatedBy oa:linking ;
                      oa:hasTarget ?target .

                  ?existingAct a prov:Activity ;
                      prov:generated ?existingAnn .

                  $statement_filter
                }
                ?existingAct prov:wasAssociatedWith / <http://www.w3.org/ns/prov#specializationOf> ?modelType .

              }
            }
            """
        )

        extra = Template(self.get_extra_inserts()).substitute(
            annotation_id=sparql_escape_uri(annotation_uri)
        )

        query_string = query_template.substitute(
            ai_graph=sparql_escape_uri(ai_graph),
            prev_annotation_uri=sparql_escape_uri(self.prev_annotation_uri),
            activity_id=activity_id_escaped,
            activity_uuid=activity_uuid_literal,
            annotation_id=sparql_escape_uri(annotation_uri),
            annotation_uuid=annotation_uuid,
            statements_uri=statements_uri,
            user=sparql_escape_uri(self.agent),
            confidence=sparql_escape_float(self.confidence),
            statement_parts=statement_parts,
            statement_filter=statement_filter,
            timing_triples=timing_triples,
            extra=extra,
        )

        try:
            update(query_string, sudo=True)
        except Exception as e:
            error_msg = (
                f"Failed to insert NamedEntityLinkingAnnotation for "
                f"{self.subject} -> {self.object}: {e}"
            )
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        return annotation_uri
