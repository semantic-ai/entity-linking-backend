import uuid
import contextlib
import time

from string import Template
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Type, TypedDict

from src.agent import SparqlResponse
from config.config import TaskOperations, settings, TaskStatus
from escape_helpers import sparql_escape_uri, sparql_escape_string
from helpers import query, update, logger
from src.utils.utils import get_prefixes_for_query, initialize_agent

class Task(ABC):
    """Base class for background tasks that process data from the triplestore."""

    def __init__(self, task_uri: str):
        super().__init__()
        self.task_uri = task_uri
        self.results_container_uris = []
        self.logger = logger
        self.agent_instance = initialize_agent()
        self.retries = 0

    @classmethod
    def supported_operations(cls) -> list[Type['Task']]:
        all_ops = []
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '__task_type__'):
                all_ops.append(subclass)
            else:
                all_ops.extend(subclass.supported_operations())
        return all_ops

    @classmethod
    def lookup(cls, task_type: str) -> Optional['Task']:
        """
        Yield all subclasses of the given class, per:
        """
        for subclass in cls.supported_operations():
            if hasattr(subclass, '__task_type__') and subclass.__task_type__ == task_type:
                return subclass
        return None

    @classmethod
    def from_uri(cls, task_uri: str) -> 'Task':
        """Create a Task instance from its URI in the triplestore."""
        q = Template(
            get_prefixes_for_query("adms", "task") +
            """
            SELECT ?task ?taskType WHERE {
              BIND($uri AS ?task)
              ?task task:operation ?taskType .
            }
        """).substitute(uri=sparql_escape_uri(task_uri))
        for b in query(q, sudo=True).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError(
                "Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str, results_container_uris: list = []) -> None:
        """Update the task status in the triplestore."""

        # Update the task status
        status_query = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + settings.default_graph + """> {
                ?task adms:status $old_status .
            }
            }
            INSERT {
            GRAPH <""" + settings.default_graph + """> {
                ?task adms:status $new_status .
            }
            }
            WHERE {
            GRAPH <""" + settings.default_graph + """> {
                BIND($task AS ?task)
                OPTIONAL { ?task adms:status $old_status . }
            }
            }
            """
        )
        query_string = status_query.substitute(
            new_status=sparql_escape_uri(new_state),
            old_status=sparql_escape_uri(old_state),
            task=sparql_escape_uri(self.task_uri)
        )

        update(query_string, sudo=True)

        # Batch-insert results containers (if any)
        if results_container_uris:
            BATCH_SIZE = 50
            insert_template = Template(
                get_prefixes_for_query("task", "adms") +
                """
                INSERT {
                GRAPH <""" + settings.default_graph + """> {
                    ?task $results_container_line .
                }
                }
                WHERE {
                    BIND($task AS ?task)
                }
                """
            )

            for i in range(0, len(results_container_uris), BATCH_SIZE):
                batch_uris = results_container_uris[i:i + BATCH_SIZE]
                results_container_line = " ;\n".join(
                    [f"task:resultsContainer {sparql_escape_uri(uri)}" for uri in batch_uris]
                )
                query_string = insert_template.substitute(
                    task=sparql_escape_uri(self.task_uri),
                    results_container_line=results_container_line
                )
                update(query_string, sudo=True)

    @contextlib.asynccontextmanager
    async def run(self):
        """Async Context manager for task execution with state transitions."""
        try:
            self.change_state(TaskStatus.SCHEDULED.value, TaskStatus.BUSY.value)
            yield
            self.change_state(
                TaskStatus.BUSY.value,
                TaskStatus.SUCCESS.value,
                self.results_container_uris,
            )
        except Exception as e:
            logger.error(f"Error executing task {self.task_uri}: {e}")
            self.change_state(TaskStatus.BUSY.value, TaskStatus.FAILED.value)
            raise

    async def execute(self):
        """Run the task and handle state transitions."""
        logger.info(f"Starting execution of task {self.task_uri}")
        async with self.run():
            logger.info(f"Running task {self.task_uri}")
            await self.process()
            logger.info(f"Finished processing task {self.task_uri}")

    @abstractmethod
    async def process(self):
        """Process task data (implemented by subclasses)."""
        pass


class NamedEntityLinkingTask(Task, ABC):
    """
    Task that processes annotations from a Named Entity Recognition (NER) service, tries to retrieve their URI using the LLM. The result is the same annotation enriched with a skos:exactMatch to the found URI.
    """

    __task_type__ = TaskOperations.NAMED_ENTITY_LINKING.value

    class NamedEntityLinkingResult(TypedDict):
        uri: str

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def fetch_data_from_input_container(self) -> dict[str, str]:
        """
        Retrieve the recognized named entity from the task's input container,
        originating from a NER service
        
        Returns:
            Dictionary containing the annotation metadata
                Keys:
                    - "entityClass": str
                    - "entityLabel": str
                    - "location": str
        """
        q = Template(
            get_prefixes_for_query("task", "oa", "rdf", "rdfs", "dct") +
            f"""
            SELECT ?annotation ?entityClass ?entityLabel ?location WHERE {{
            GRAPH <{settings.default_graph}> {{
                $task task:inputContainer ?container .
                ?container task:hasResource ?annotation .
                ?annotation oa:hasBody ?statement .

                ?statement rdf:predicate ?predicate .
                ?statement rdf:object ?entity .

                ?entity a ?entityClass ;
                    rdfs:label ?entityLabel .

                OPTIONAL {{
                    ?entity dct:spatial ?location . 
                }}
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        logger.info(f"Fetching data for task {self.task_uri} with query: {q}")

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            return
        
        return {
            "annotation": bindings[0].get("annotation", {}).get("value"),
            "entityClass": bindings[0].get("entityClass", {}).get("value"),
            "entityLabel": bindings[0].get("entityLabel", {}).get("value"),
            "location": bindings[0].get("location", {}).get("value", "Unknown location"),
        }

    def copy_annotation(self, prev_annotation_uri: str, entity_uri: str) -> str:
        """
        Function to create a copy of an annotation and add a skos:exactMatch to the found entity URI.

        Args:
            prev_annotation_uri: URI of the previous (NER) annotation
            entity_uri: URI of the found entity to be linked to the annotation

        Returns:
            The created annotation URI.
        """
        new_annotation_uuid = str(uuid.uuid4())
        new_annotation_uri = f"http://data.lblod.info/id/annotations/{new_annotation_uuid}"

        q = Template(
            get_prefixes_for_query("eli", "mu", "skos", "oa")
            + f"""
            INSERT {{
            GRAPH <{settings.default_graph}> {{
                $new_annotation a oa:Annotation ;
                    mu:uuid $new_uuid ;
                    ?p ?o .

                ?statement ?pS ?oS .
                ?entity ?pE ?oE .
                ?entity skos:exactMatch $entity_uri .
            }}
            }}
            WHERE {{
            GRAPH <{settings.default_graph}> {{
                BIND($prev_annotation_uri AS ?prevAnnotation)
                ?prevAnnotation ?p ?o .

                ?prevAnnotation oa:hasBody ?statement .
                ?statement ?pS ?oS .

                ?statement rdf:object ?entity .
                ?entity ?pE ?oE .
            }}
            }}
            """
        ).substitute(
            prev_annotation_uri=sparql_escape_uri(prev_annotation_uri),
            new_annotation=sparql_escape_uri(new_annotation_uri),
            new_uuid=sparql_escape_string(new_annotation_uuid),
            entity_uri=sparql_escape_uri(entity_uri),
        )

        update(q, sudo=True)

        return new_annotation_uri

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
            GRAPH <{settings.default_graph}> {{
                $container a nfo:DataContainer ;
                    mu:uuid "$uuid" ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=container_id,
            resource=sparql_escape_uri(resource)
        )

        update(q, sudo=True)
        return container_uri

    async def process(self):
        """
        Implementation of Task's process function that
         - retrieves the recognized Named Entity from the task's input data container
         - sends query to LLM to retrieve the URI of the entity based on its class, label and (optionally) location
         - creates a copy of the input annotation and adds a skos:exactMatch to the found URI
        """
        if not hasattr(self, "retries"):
            self.retries = 0

        while self.retries < settings.llm_max_retries:
            self.retries += 1
            try:
                logger.info(f"Processing task {self.task_uri} of type {self.__task_type__}")
                input = self.fetch_data_from_input_container()

                logger.info(f"Fetched input for task {self.task_uri}: {input}")

                logger.info(f"Sending query to LLM for task {self.task_uri} with entity class {input['entityClass']} and entity label {input['entityLabel']} and location {input['location']}")
                
                response: SparqlResponse = await self.agent_instance.run_sparql_request_structured(
                    entity_class=input["entityClass"],
                    entity_label=input["entityLabel"],
                    location=input["location"]
                )
                results = response.results
                logger.info(f"Received result from LLM for task {self.task_uri}: {results}")

                if len(results) > 0 and results[0].uri:
                    logger.info(f"Copying annotation and linking to found URI {results[0].uri} for task {self.task_uri}")
                    new_annotation = self.copy_annotation(prev_annotation_uri=input["annotation"], entity_uri=results[0].uri)
                    logger.info(f"Successfully processed task {self.task_uri}, creating output container for result")
                    self.results_container_uris.append(self.create_output_container(resource=new_annotation))
                    logger.info(f"Finished creating output container for task {self.task_uri}")

            except Exception as e:
                logger.error(f"Error processing task {self.task_uri}: {e}")
                if self.retries >= settings.llm_max_retries:
                    logger.error(f"Max retries reached for task {self.task_uri}. Failing task.")
                    raise
                else:
                    logger.info(f"Retrying task {self.task_uri} (attempt {self.retries}/{settings.llm_max_retries})")
                    time.sleep(5)  # Wait before retrying