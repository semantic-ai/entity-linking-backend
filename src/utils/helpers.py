
import uuid
import datetime
import logging
import os
import sys
from rdflib.namespace import DC
from .escape_helpers import sparql_escape
from SPARQLWrapper import SPARQLWrapper, JSON

# =====================
# Environment Variables
# =====================

# LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

# MODE: Deployment mode (development, production). Defaults to production.
MODE = os.environ.get('MODE', 'production')

# MU_SPARQL_ENDPOINT: SPARQL endpoint URL
MU_SPARQL_ENDPOINT = os.environ.get('MU_SPARQL_ENDPOINT', 'http://localhost:8890/sparql')

# MU_APPLICATION_GRAPH: Graph in the triple store
MU_APPLICATION_GRAPH = os.environ.get('MU_APPLICATION_GRAPH', 'http://mu.semte.ch/graphs/public')

# MU_SPARQL_TIMEOUT: Timeout for SPARQL queries (in seconds)
MU_SPARQL_TIMEOUT = os.environ.get('MU_SPARQL_TIMEOUT')

# SPARQL Query Logging
# LOG_SPARQL_ALL: Log all queries (default True)
LOG_SPARQL_ALL = os.environ.get('LOG_SPARQL_ALL', 'true')
# LOG_SPARQL_QUERIES: Log read queries (overrides LOG_SPARQL_ALL)
LOG_SPARQL_QUERIES = os.environ.get('LOG_SPARQL_QUERIES', LOG_SPARQL_ALL).lower() == 'true'
# LOG_SPARQL_UPDATES: Log update queries (overrides LOG_SPARQL_ALL)
LOG_SPARQL_UPDATES = os.environ.get('LOG_SPARQL_UPDATES', LOG_SPARQL_ALL).lower() == 'true'

# =====================

"""
The template provides the user with several helper methods. They aim to give you a step ahead for:

- logging
- JSONAPI-compliancy
- SPARQL querying

The below helpers can be imported from the `helpers` module. For example:
```py
from helpers import *
```

Available functions:
"""


# TODO: Figure out how logging works when production uses multiple workers
log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
log_dir = '/logs'
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = logging.getLogger('MU_PYTHON_TEMPLATE_LOGGER')
logger.setLevel(log_levels.get(LOG_LEVEL, logging.INFO))
fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, 'logs'))
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(stream=sys.stdout)  # or stderr?
logger.addHandler(consoleHandler)

def generate_uuid():
    """Generates a random unique user id (UUID) based on the host ID and current time"""
    return str(uuid.uuid1())


def log(msg, *args, **kwargs):
    """
    Write a log message to the log file.
    
    Works exactly the same as the logging.info (https://docs.python.org/3/library/logging.html#logging.info) method from pythons' logging module.
    Logs are written to the /logs directory in the docker container.  
    
    Note that the `helpers` module also exposes `logger`, which is the logger instance (https://docs.python.org/3/library/logging.html#logger-objects) 
    used by the template. The methods provided by this instance can be used for more fine-grained logging.
    """
    return logger.info(msg, *args, **kwargs)






sparqlQuery = SPARQLWrapper(MU_SPARQL_ENDPOINT, returnFormat=JSON)
sparqlUpdate = SPARQLWrapper(os.environ.get('MU_SPARQL_UPDATEPOINT', MU_SPARQL_ENDPOINT), returnFormat=JSON)
sparqlUpdate.method = 'POST'
if MU_SPARQL_TIMEOUT:
    timeout = int(MU_SPARQL_TIMEOUT)
    sparqlQuery.setTimeout(timeout)
    sparqlUpdate.setTimeout(timeout)

def query(the_query):
    """Execute the given SPARQL query (select/ask/construct) on the triplestore and returns the results in the given return Format (JSON by default)."""
    sparqlQuery.setQuery(the_query)
    if LOG_SPARQL_QUERIES:
        log("Execute query: \n" + the_query)
    try:
        return sparqlQuery.query().convert()
    except Exception as e:
        log("Failed Query: \n" + the_query)
        raise e


def update(the_query):
    """Execute the given update SPARQL query on the triplestore. If the given query is not an update query, nothing happens."""
    sparqlUpdate.setQuery(the_query)
    if sparqlUpdate.isSparqlUpdateRequest():
        if LOG_SPARQL_UPDATES:
            log("Execute query: \n" + the_query)
        try:
            sparqlUpdate.query()
        except Exception as e:
            log("Failed Query: \n" + the_query)
            raise e


def update_modified(subject, modified=datetime.datetime.now()):
    """(DEPRECATED) Executes a SPARQL query to update the modification date of the given subject URI (string).
     The default date is now."""
    query = " WITH <%s> " % MU_APPLICATION_GRAPH
    query += " DELETE {"
    query += "   < %s > < %s > %s ." % (subject, DC.Modified, sparql_escape(modified))
    query += " }"
    query += " WHERE {"
    query += "   <%s> <%s> %s ." % (subject, DC.Modified, sparql_escape(modified))
    query += " }"
    update(query)

    query = " INSERT DATA {"
    query += "   GRAPH <%s> {" % MU_APPLICATION_GRAPH
    query += "     <%s> <%s> %s ." % (subject, DC.Modified, sparql_escape(modified))
    query += "   }"
    query += " }"
    update(query)