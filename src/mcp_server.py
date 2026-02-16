import argparse
import json
import logging
import asyncio
import warnings

# Aggressively suppress warnings
warnings.simplefilter("ignore")

from fastmcp import FastMCP, settings as mcp_settings

from config.config import settings, endpoints
from src.knowledge_base import get_knowledge_base
from src.utils.utils import format_docs

from sparql_llm.utils import get_prefixes_and_schema_for_endpoints
from sparql_llm.validate_sparql import validate_sparql

from typing_extensions import Required
from typing import Optional

from src.tools.sparql_search import SparqlClient
from src.tools.nominatim_search import NominatimGeocoder
from src.tools.web_search import DuckDuckGoSearch

# Setup Logger
from helpers import logger

# Prefixes and schema for validation
prefixes_map, endpoints_void_dict = get_prefixes_and_schema_for_endpoints(endpoints)

mcp = FastMCP(
    name="Decide MCP Server",
    instructions="Provide tools that help users access Linked Data resources (SPARQL), general web info, and location services.",
)

# Initialize Knowledge Base
knowledge_base = get_knowledge_base()
try:
    knowledge_base.initialize()
except Exception as e:
    logger.error(f"Error checking or initializing knowledge base: {e}")

# --- Legacy Tools ---

@mcp.tool()
async def search_location(query: str, city: Optional[str] = "Gent", country: str = "BE") -> str:
    """
    Search for a location (entity linking) based on a query, city, and country.
    Returns the geocoded result including address and coordinates.

    Args:
        query (str): The location query string.
        city (Optional[str]): The city to narrow down the search.
        country (str): The country to narrow down the search.
    Returns:
        str: JSON string of the geocoding result containing OpenStreetMap URI, address, latitude, and longitude.
    """
    geocoder = NominatimGeocoder(base_url=settings.nominatim_endpoint)
    result = await geocoder.search(query=query, city=city, country=country)
    return json.dumps(result) if result else "No results found"

@mcp.tool()
async def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for additional information. Useful for general knowledge questions about persons, places, events, etc.
    Args:
        query (str): The search query string.
        max_results (int): Maximum number of results to return.
    Returns:
        str: JSON string of the search results.
    """
    
    ddg_search = DuckDuckGoSearch()
    results = await ddg_search.search(query=query, max_results=max_results)
    if results:
        return json.dumps(results)
    return "No results found."


# --- SPARQL Tools ---

PROMPT_TOOL_SPARQL = """Formulate a precise SPARQL query to access specific linked data resources and answer the user's question.

## SPARQL Query Guidelines
- **Always include the endpoint URL** as a comment at the start: `#+ endpoint: http://example.org/sparql`
- **Use only ONE endpoint** per query
- **Base your query on the provided context** - never create generic or unsupported queries
- **Use appropriate prefixes** and class names from the schema documentation

## Knowledge Base
The following {docs_count} documents contain relevant query examples, and classes schemas to help you construct an accurate response:

{formatted_docs}
"""

@mcp.tool()
async def search_sparql_docs(question: str, potential_classes: list[str], steps: list[str]) -> str:
    """Assist users in writing SPARQL queries to access resources by retrieving relevant examples and classes schema.
    
    Args:
        question: The question to be answered with a SPARQL query
        potential_classes: High level concepts and potential classes that could be found in the SPARQL endpoints
        steps: Split the question in standalone smaller parts if relevant

    Returns:
        A formatted string containing relevant SPARQL examples and classes schema to help construct the SPARQL query.
    """
    def _sync_search():
        relevant_docs = knowledge_base.search(question, potential_classes, steps)
        return PROMPT_TOOL_SPARQL.format(docs_count=str(len(relevant_docs)), formatted_docs=format_docs(relevant_docs))

    return await asyncio.to_thread(_sync_search)


FIX_QUERY_PROMPT = """Please fix the query, and try again.
We suggest you to make the query less restricted, e.g. use a broader regex for string matching instead of exact match,
ignore case, make sure you are not overriding an existing variable with BIND, or break down your query in smaller parts
and check them one by one."""

@mcp.tool()
async def execute_sparql_query(sparql_query: str, endpoint_url: str) -> str:
    """Execute a SPARQL query against a SPARQL endpoint.

    Args:
        sparql_query: A valid SPARQL query string
        endpoint_url: The SPARQL endpoint URL to execute the query against

    Returns:
        The query results in JSON format
    """
    resp_msg = ""
    # First check if query valid based on classes schema and known prefixes
    validation_output = await asyncio.to_thread(validate_sparql, sparql_query, endpoint_url, prefixes_map, endpoints_void_dict)
    if validation_output["fixed_query"]:
        # Pass the fixed query to the client
        resp_msg += f"Fixed the prefixes of the generated SPARQL query automatically:\n```sparql\n{validation_output['fixed_query']}\n```\n"
        sparql_query = validation_output["fixed_query"]
    if validation_output["errors"]:
        # Recall the LLM to try to fix the errors
        error_str = "- " + "\n- ".join(validation_output["errors"])
        resp_msg += (
            "The query generated in the original response is not valid according to the endpoints schema.\n"
            f"### Validation results\n{error_str}\n"
            f"### Erroneous SPARQL query\n```sparql\n{validation_output['original_query']}\n```\n"
            "Fix the SPARQL query helping yourself with the error message and context from previous messages."
        )
        return resp_msg
    # Execute the SPARQL query
    try:
        client = SparqlClient(endpoint=endpoint_url)
        # SparqlClient.search returns List[Dict] (the bindings) or raises exception
        bindings = await client.search(query=sparql_query, max_results=50) 
        
        if not bindings:
            # If no results, return a message to ask fix the query
            resp_msg += f"SPARQL query returned no results. {FIX_QUERY_PROMPT}\n```sparql\n{sparql_query}\n```"
        else:
            # If results, return them (limit to first 50 rows if too many)
            resp_msg += f"Results of SPARQL query execution on {endpoint_url}"
            if len(bindings) > 50:
                bindings = bindings[:50]
                resp_msg += f" (showing first 50 results)"
            
            # Construct a response object 
            res = {"results": {"bindings": bindings}}
            resp_msg += f":\n```\n{json.dumps(res, indent=2)}\n```"
    except Exception as e:
        resp_msg += f"SPARQL query returned error: {e}. {FIX_QUERY_PROMPT}\n```sparql\n{sparql_query}\n```"
    return resp_msg


def cli() -> None:
    """Run the MCP server with appropriate transport."""
    parser = argparse.ArgumentParser(
        description="A Model Context Protocol (MCP) server for Decide project."
    )
    parser.add_argument("--http", action="store_true", help="Use Streamable HTTP transport")
    parser.add_argument("--port", type=int, default=7000, help="Port to run the server on")
    args = parser.parse_args()

    if args.http:
        # Update global FastMCP settings
        mcp_settings.port = args.port
        mcp_settings.log_level = "INFO"
        mcp_settings.debug = True
        
        # Pass settings to run just in case, or rely on global settings if supported
        try:
             mcp.run(transport="sse", port=args.port)
        except TypeError:
             # Fallback if port is not accepted in run
             mcp.run(transport="sse")
    else:
        mcp.run()

if __name__ == "__main__":
    cli()
