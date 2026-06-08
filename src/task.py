import uuid
import time

from datetime import datetime, timezone
from string import Template
from typing import TypedDict

from src.config import settings, endpoints
from escape_helpers import sparql_escape_uri, sparql_escape_string
from helpers import query, update, logger
from decide_ai_service_base.sparql_config import (
    AGENT_TYPES,
    AI_COMPONENTS,
    GRAPHS,
    JOB_STATUSES,
    get_prefixes_for_query,
)
from decide_ai_service_base.task import DecisionTask

from src.linkers import LINKERS
from src.nel_annotation import NamedEntityLinkingAnnotation


class NamedEntityLinkingTask(DecisionTask):
    """
    Task that processes annotations from a Named Entity Recognition (NER) service, tries to retrieve their URI using the LLM. The result is the same annotation enriched with a skos:exactMatch to the found URI.
    """

    # @todo integrate in ai service base
    __task_type__ = "http://lblod.data.gift/id/jobs/concept/TaskOperation/named-entity-linking"

    class NamedEntityLinkingResult(TypedDict):
        uri: str

    def execute(self):
        """Atomic SCHEDULED→BUSY claim before delegating to the base lifecycle.

        Two parallel TaskProcessor invocations (e.g. /delta during startup drain)
        can both see the same SCHEDULED task. The base's change_state is not atomic
        on the prior state, so without this guard the task could be executed twice.
        """
        if not self._claim_busy_atomically():
            logger.info(f"Task {self.task_uri} already claimed by another worker; skipping")
            return
        super().execute()

    def _claim_busy_atomically(self) -> bool:
        claim_marker = str(uuid.uuid4())
        update_q = Template(
            get_prefixes_for_query("adms", "ext")
            + """
            DELETE { GRAPH $graph { $task adms:status $scheduled . } }
            INSERT {
                GRAPH $graph {
                    $task adms:status $busy ;
                          ext:claimedBy $marker .
                }
            }
            WHERE { GRAPH $graph { $task adms:status $scheduled . } }
            """
        ).substitute(
            graph=sparql_escape_uri(GRAPHS["jobs"]),
            task=sparql_escape_uri(self.task_uri),
            scheduled=sparql_escape_uri(JOB_STATUSES["scheduled"]),
            busy=sparql_escape_uri(JOB_STATUSES["busy"]),
            marker=sparql_escape_string(claim_marker),
        )
        update(update_q, sudo=True)

        check_q = Template(
            get_prefixes_for_query("ext")
            + """
            ASK { GRAPH $graph { $task ext:claimedBy $marker . } }
            """
        ).substitute(
            graph=sparql_escape_uri(GRAPHS["jobs"]),
            task=sparql_escape_uri(self.task_uri),
            marker=sparql_escape_string(claim_marker),
        )
        return query(check_q, sudo=True).get("boolean", False)

    def fetch_governing_unit_uri(self) -> str:
        """
        Retrieve the governing unit URI provided in the input container
        of the first task in the same job as this task.

        Returns:
            String containing the governing unit URI or
            "Unknown URI" in case no governing unit was provided.
        """
        governing_unit_uri = "Unknown URI"

        q = Template(f"""
            {get_prefixes_for_query("task", "dct", "nfo", "nie")}
            SELECT ?resource WHERE {{
                GRAPH $graph {{
                    $task dct:isPartOf ?job .
                    ?firstTask dct:isPartOf ?job ;
                            task:index "0" ;
                            task:inputContainer ?container .
                    ?container task:hasResource ?resource .
                }}
            }}
        """).substitute(
            graph=sparql_escape_uri(GRAPHS["jobs"]),
            task=sparql_escape_uri(self.task_uri)
        )

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if bindings:
            governing_unit_uri = bindings[0]["resource"]["value"]

        return governing_unit_uri
    

    def fetch_governing_unit_name(self, governing_unit_uri: str) -> str:
        """
        Retrieve the name of the governing unit based on its URI.

        Args:
            governing_unit_uri: String containing the URI of the governing unit

        Returns:
            String containing the name of the governing unit or
            "Unknown name" in case the name could not be retrieved.
        """
        governing_unit_name = "Unknown name"

        q = Template(f"""
            {get_prefixes_for_query("skos")}
            SELECT ?name WHERE {{
                $governing_unit skos:prefLabel ?name .
            }} LIMIT 1
        """).substitute(
            governing_unit=sparql_escape_uri(governing_unit_uri)
        )

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if bindings:
            governing_unit_name = bindings[0]["name"]["value"]

        return governing_unit_name
    

    def fetch_data_from_input_container(self) -> dict[str, str]:
        """
        Retrieve the recognized named entity by bridging the harvesting graph 
        with the actual data graph.
        """
        q = Template(
            get_prefixes_for_query("task", "oa", "rdf", "rdfs", "dct", "eli") +
            f"""
            SELECT ?annotation ?entity ?entityClass ?entityLabel ?location WHERE {{
                $task dct:isPartOf ?job .
                {{
                    $task task:inputContainer ?container .
                    ?container task:hasResource ?annotation .
                }}
                UNION
                {{
                    ?job <http://mu.semte.ch/vocabularies/ext/shapeForTargets> / <http://www.w3.org/ns/shacl#targetNode> ?expression.
                    ?expression a eli:Expression.
                    ?annotation oa:hasTarget / oa:hasSource ?expression .
                    FILTER NOT EXISTS {{
                        ?original <http://purl.org/linguistics/gold/translation> ?expression .
                    }}

                    FILTER NOT EXISTS {{
                        ?splitter dct:isPartOf ?job .
                        ?splitter task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/annotation-split-tasks> .
                    }}
                }}
                
                GRAPH $publication_graph {{
                    ?annotation oa:motivatedBy oa:linking .
                    ?annotation oa:hasBody ?statement .
                    ?statement rdf:object ?entity .

                    ?entity a ?entityClass ;
                            rdfs:label ?entityLabel .

                    OPTIONAL {{
                        ?entity dct:spatial ?location . 
                    }}
                }}
            }}
            """
        ).substitute(
            task=sparql_escape_uri(self.task_uri),
            default_graph=sparql_escape_uri(GRAPHS["jobs"]),
            publication_graph=sparql_escape_uri(GRAPHS["ai"])
        )

        logger.info(f"Fetching data for task {self.task_uri} with query: {q}")

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            return

        results = [
            {
                "annotation": b.get("annotation", {}).get("value"),
                "entityClass": b.get("entityClass", {}).get("value"),
                "entityLabel": b.get("entityLabel", {}).get("value"),
                "location": b.get("location", {}).get("value", "Unknown location"),
                "entity": b.get("entity", {}).get("value"),
            }
            for b in bindings if not "person" in b.get("entityClass", {}).get("value", "").lower() # Excluding mandataries for now
        ]

        return results

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container with a resource

        Args:
            resource: String containing an URI of a resource that should be added to the container with the task:hasResource property

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH $graph {{
                $container a nfo:DataContainer ;
                    mu:uuid "$uuid" ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=container_id,
            resource=sparql_escape_uri(resource),
            graph=sparql_escape_uri(GRAPHS["jobs"])
        )

        update(q, sudo=True)
        return container_uri

    def _resolve_location(self, input: dict) -> str:
        """Pick the location string for the linker.

        Falls back to the governing-unit name when the upstream NER step
        did not attach a `dct:spatial` value.
        """
        if input["location"] == "Unknown location":
            governing_unit_uri = self.fetch_governing_unit_uri()
            return [self.fetch_governing_unit_name(governing_unit_uri), governing_unit_uri]
        return input["location"]

    def process(self):
        """
        Implementation of Task's process function that
         - retrieves the recognized Named Entity from the task's input data container
         - dispatches to a deterministic linker (per src.linkers.LINKERS) keyed on entityClass
         - creates a new NEL annotation referencing the original, adds skos:exactMatch and provenance
        """
        inputs = self.fetch_data_from_input_container()

        if inputs is None:
            return

        for input in inputs:
            self.retries = 0
            success = False
            while not success and self.retries < settings.llm_max_retries:
                self.retries += 1
                try:
                    logger.info(
                        f"Processing task {self.task_uri} of type {self.__task_type__}"
                    )
                    logger.info(f"Fetched input for task {self.task_uri}: {input}")

                    entity_class = input["entityClass"]
                    linker = LINKERS.get(entity_class.strip().lower())
                    if linker is None:
                        logger.warning(
                            f"No linker registered for entity_class {entity_class!r}; "
                            f"skipping entity {input.get('entity')!r}"
                        )
                        success = True
                        break

                    [location, location_uri] = self._resolve_location(input)

                    logger.info(
                        f"Linking {input['entityLabel']!r} ({entity_class}) "
                        f"in {location!r} via {type(linker).__name__}"
                    )

                    start_time = datetime.now(timezone.utc)
                    result = linker.link(
                        entity_label=input["entityLabel"],
                        location=location,
                        location_uri=location_uri,
                        subject_uri=input.get("entity"),
                    )
                    end_time = datetime.now(timezone.utc)

                    if result is None:
                        logger.info(
                            f"No link produced for {input.get('entity')!r} "
                            f"({entity_class}); not creating an annotation"
                        )
                        success = True
                        break

                    logger.info(
                        f"Creating NEL annotation linking {input['entity']!r} "
                        f"to {result.uri!r} for task {self.task_uri}"
                    )
                    activity_uri = f"http://data.lblod.info/id/activities/{uuid.uuid4()}"
                    nel = NamedEntityLinkingAnnotation(
                        prev_annotation_uri=input["annotation"],
                        subject_uri=input["entity"],
                        object_uri=result.uri,
                        activity_id=activity_uri,
                        source_uri=input["entity"],
                        agent=AI_COMPONENTS["linker"],
                        agent_type=AGENT_TYPES.get("linker", ""),
                        start_time=start_time,
                        end_time=end_time,
                        extra_triples=result.extra_triples,
                    )
                    new_annotation = nel.add_to_triplestore_if_not_exists()
                    self.results_container_uris.append(
                        self.create_output_container(resource=new_annotation)
                    )
                    logger.info(
                        f"Finished creating output container for task {self.task_uri}"
                    )

                    success = True
                except Exception as e:
                    logger.error(f"Error processing task {self.task_uri}: {e}")
                    if self.retries >= settings.llm_max_retries:
                        logger.error(
                            f"Max retries reached for task {self.task_uri}. Failing task."
                        )
                    else:
                        logger.info(
                            f"Retrying task {self.task_uri} "
                            f"(attempt {self.retries}/{settings.llm_max_retries})"
                        )
                        time.sleep(5)
