import asyncio
from string import Template
import time

from escape_helpers import sparql_escape_uri
from helpers import logger, update, query

from config.config import TaskOperations, settings, TaskStatus
from src.utils.utils import get_prefixes_for_query, wait_for_triplestore
from src.task import Task

############################################################
# TODO: keep this generic and extract into packaged module later
############################################################

class TaskNotFoundException(Exception):
    "Raised when task is not found"
    pass

async def startup_tasks():
    wait_for_triplestore()
    # on startup fail existing busy tasks
    logger.info(f"Failing busy tasks open tasks...")
    fail_busy_tasks()
    logger.info(f"Processing open tasks...")
    await asyncio.sleep(5)  # Give the MCP server time to be ready
    await process_open_tasks()
    logger.info(f"Processing open tasks finished")

def fail_busy_tasks():
    logger.info("Startup: failing busy tasks if there are any")
    update(f"""
      PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
      PREFIX dct: <http://purl.org/dc/terms/>
      PREFIX adms: <http://www.w3.org/ns/adms#>
      PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
      DELETE {{
        GRAPH {sparql_escape_uri(settings.default_graph)} {{
            ?task  adms:status ?status
        }}
      }}
      INSERT {{
        GRAPH {sparql_escape_uri(settings.default_graph)} {{
          ?task adms:status {sparql_escape_uri(TaskStatus.FAILED.value)}
        }}
      }}
      WHERE  {{
        GRAPH {sparql_escape_uri(settings.default_graph)} {{
            ?task a task:Task .
            ?task dct:isPartOf ?job;
            task:operation ?operation ;
            adms:status ?status.
        VALUES ?operation {{
        {sparql_escape_uri(TaskOperations.NAMED_ENTITY_LINKING.value)}
        }}
        VALUES ?status {{
        {sparql_escape_uri(TaskStatus.BUSY.value)}
        }}
        }}
      }}

        """, sudo=True)

def load_task(subject, graph = settings.default_graph):
    query_template = Template("""
  PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
  PREFIX dct: <http://purl.org/dc/terms/>
  PREFIX adms: <http://www.w3.org/ns/adms#>
  PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
  SELECT DISTINCT ?id ?job ?jobId ?created ?modified ?status ?index ?operation ?error WHERE {
      GRAPH $graph {
        $subject a task:Task .
        $subject dct:isPartOf ?job;
                      mu:uuid ?id;
                      dct:created ?created;
                      dct:modified ?modified;
                      adms:status ?status;
                      task:index ?index;
                      task:operation ?operation.
        ?job mu:uuid ?jobId.
        OPTIONAL { $subject task:error ?error. }
      }
    }

    """)

    query_string = query_template.substitute(
        graph = sparql_escape_uri(graph),
        subject = sparql_escape_uri(subject)
    )

    results = query(query_string, sudo=True)
    bindings = results["results"]["bindings"]
    if len(bindings) == 1:
        item = bindings[0]
        id = item['id']['value']
        job = item['job']['value']
        job_id = item['jobId']['value']
        status = item['status']['value']
        index = item['index']['value']
        operation = item['operation']['value']
        error = item.get('error', {}).get('value', None)
        return {
            'id': id,
            'job': job,
            'job_id' : job_id,
            'status': status,
            'operation': operation,
            'index': index,
            'error': error,
            'uri': subject
        }
    elif len(bindings) == 0:
        raise TaskNotFoundException()
    else:
        raise Exception(f"Unexpected result loading task: {results}")


async def process_open_tasks():
    logger.info("Checking for open tasks...")
    uri = get_one_open_task()
    while uri is not None:
        logger.info(f"Processing {uri}")
        task = Task.from_uri(uri)
        logger.info(f"Loaded task {task.task_uri}")
        await task.execute()
        logger.info(f"Finished processing {uri}")
        await asyncio.sleep(5)
        uri = get_one_open_task()

def get_one_open_task() -> str | None:
    q = f"""
        {get_prefixes_for_query("task", "adms")}
        SELECT ?task WHERE {{
        GRAPH <{settings.default_graph}> {{
            ?task adms:status <{TaskStatus.SCHEDULED.value}> ;
                  task:operation ?operation .
            VALUES ?operation {{
              {sparql_escape_uri(TaskOperations.NAMED_ENTITY_LINKING.value)}
            }}
        }}
        }}
        limit 1
    """
    results = query(q, sudo=True)
    bindings = results.get("results", {}).get("bindings", [])
    return bindings[0]["task"]["value"] if bindings else None