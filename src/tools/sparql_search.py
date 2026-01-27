import logging
import re
from typing import List, Dict, Any, Optional
import httpx

# Configure logging (usually done at the app level)
logger = logging.getLogger(__name__)

class SparqlClient:
    """
    A client to perform SPARQL searches against a specific endpoint.
    """

    def __init__(self, endpoint: str):
        """
        Initialize the SPARQL client.

        Args:
            endpoint (str): The SPARQL endpoint URL.
        """
        self.endpoint = endpoint

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """
        Removes comments from a SPARQL query while preserving # inside strings/IRIs.
        Handles escaped quotes and multi-line strings via Regex.
        """
        # Pattern explanation:
        # 1. ("""...""") or ('''...'''): Multi-line strings
        # 2. ("...") or ('...'): Single-line strings (handling escaped quotes)
        # 3. (<...>): IRIs
        # 4. (#.*): Comments (what we want to remove)
        pattern = re.compile(
            r'("""(?:.|\n)*?""")|' 
            r"('''(?:.|\n)*?''')|" 
            r'("(?:\\.|[^"\\])*")|' 
            r"('(?:\\.|[^'\\])*')|" 
            r'(<[^>]*>)|' 
            r'(#.*)', 
            re.UNICODE
        )

        def replacer(match):
            # If the 6th group (comment) is matched, return empty string.
            # Otherwise, return the match (string or IRI) as is.
            if match.group(6):
                return ""
            return match.group(0)

        cleaned = pattern.sub(replacer, query)
        return "\n".join([line for line in cleaned.splitlines() if line.strip()])

    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform a SPARQL search.

        Args:
            query (str): The SPARQL query string.
            max_results (int, optional): Truncate results list to this length. 
                                         Note: Use SPARQL 'LIMIT' for server-side optimization.

        Returns:
            List[Dict[str, Any]]: A list of binding dictionaries.
        
        Raises:
            RuntimeError: If the SPARQL query fails.
        """
        clean_query = self._sanitize_query(query)
        logger.debug(f"Executing SPARQL Query:\n{clean_query}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.endpoint, 
                    params={"query": clean_query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json"},
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
            
            bindings = data.get("results", {}).get("bindings", [])
            
            if max_results:
                return bindings[:max_results]
            return bindings

        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            raise RuntimeError(f"SPARQL query failed on {self.endpoint}") from e

