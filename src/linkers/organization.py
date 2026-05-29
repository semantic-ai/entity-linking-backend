from typing import Optional
import requests

from helpers import logger
from escape_helpers import sparql_escape_uri, sparql_escape_float

from src.linkers.base import EntityLinker, LinkerResult
from src.config import settings

class OrganizationLinker(EntityLinker):
    """Resolve an organization label to a URI via vector search.

    Elasticsearch vector search on the org
    name field, with `location` used as a filter field. Returns the URI of
    the best-scoring document above a configurable similarity threshold,
    or None when no candidate is confident enough.

    """

    def link(
        self,
        entity_label: str,
        location: str,
        subject_uri: Optional[str] = None,
    ) -> Optional[LinkerResult]:

        url = f"{settings.search_endpoint}/organizations/search/"
        params = {
            "filter[name.*]": entity_label,
            "filter[:term:owning-body]": location,
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                logger.error(
                    "OrganizationLinker received non-dict JSON payload",
                    extra={"payload_type": type(payload).__name__},
                )
                return None

            results = payload.get("data") or []
            if not results:
                logger.info(
                    "OrganizationLinker found no matching organizations",
                    extra={"entity_label": entity_label, "location": location},
                )
                return None

            first_org = results[0]
            uri = first_org.get("attributes", {}).get("uri")
            if not uri:
                logger.error(
                    "OrganizationLinker response missing uri for first organization",
                    extra={"first_org": first_org},
                )
                return None
            extra_triples = None
            score = first_org.get("score", None)
            if score:
                extra_triples = f"{sparql_escape_uri(subject_uri)} <http://mu.semte.ch/vocabularies/ext/muSearchScore> {sparql_escape_float(score)} ."

            return LinkerResult(uri=uri, extra_triples=extra_triples)
        except requests.RequestException as e:
            logger.error(
                f"OrganizationLinker HTTP request failed: {e} "
                f"entity_label={entity_label!r}, location={location!r}"
            )
            return None
        except ValueError as e:
            logger.error(
                f"OrganizationLinker failed to decode JSON response: {e}"
            )
            return None
