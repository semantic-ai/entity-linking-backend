import asyncio
from typing import Optional

from helpers import logger

from src.config import settings
from src.linkers.base import EntityLinker, LinkerResult
from src.tools.nominatim_search import NominatimGeocoder
from src.utils.nominatim_parser import NominatimParser


class LocationLinker(EntityLinker):
    """Resolve a location label deterministically via Nominatim.

    """

    def __init__(self):
        self._geocoder = NominatimGeocoder(base_url=settings.nominatim_endpoint)
        self._parser = NominatimParser()

    def link(
        self,
        entity_label: str,
        location: str,
        location_uri: str,
        subject_uri: Optional[str] = None,
    ) -> Optional[LinkerResult]:
        if not entity_label:
            return None

        city = location if location and location != "Unknown location" else None

        result = self._geocoder.search(query=entity_label, city=city)
        if not result:
            logger.info(
                f"Nominatim returned no result for {entity_label!r} (city={city!r})"
            )
            return None

        osm_url = result.get("osm_url")
        if not osm_url:
            return None

        extra_triples = ""
        if "openstreetmap.org" in osm_url:
            try:
                parts = osm_url.rstrip("/").split("/")
                osm_type, osm_id = parts[-2], parts[-1]
                lookup_result = self._geocoder.lookup_osm(osm_type, osm_id)
                if lookup_result:
                    extracted = self._parser.detect_and_extract(lookup_result)
                    extra_triples = self._parser.format_triples(
                        extracted, subject_uri=subject_uri
                    )
            except Exception as e:
                logger.error(
                    f"Failed to enrich Nominatim result for {osm_url}: {e}"
                )

        return LinkerResult(uri=osm_url, extra_triples=extra_triples)
