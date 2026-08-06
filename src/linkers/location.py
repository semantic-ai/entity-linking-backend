import asyncio
from typing import Optional

from helpers import logger

from src.config import settings, location_overrides
from src.linkers.base import EntityLinker, LinkerResult
from src.tools.nominatim_search import NominatimGeocoder
from src.utils.nominatim_parser import NominatimParser


class LocationLinker(EntityLinker):
    """Resolve a location label deterministically via Nominatim.

    Labels matching a `location_overrides` entry in config.json (keyed by
    entity label, case-insensitive) skip Nominatim search entirely. This is
    for pilot cities where free-text search ranks the wrong POI (e.g. a
    station instead of the city hall) - see README for how to add one.
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

        override = location_overrides.get(entity_label.strip().casefold())
        if override:
            return self._link_override(override, subject_uri)

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

        return LinkerResult(uri=osm_url, extra_triples=self._enrich(osm_url, subject_uri))

    def _link_override(
        self, override: dict, subject_uri: Optional[str]
    ) -> Optional[LinkerResult]:
        uri = override.get("uri")
        if not uri:
            logger.warning(f"location_overrides entry missing 'uri': {override!r}")
            return None

        if "lat" in override and "lon" in override:
            extracted = {
                "wkt": self._parser.geojson_to_wkt(
                    {"lat": override["lat"], "lon": override["lon"]}
                ),
                "label": override.get("label"),
                "exact_match": uri,
            }
            extra_triples = self._parser.format_triples(extracted, subject_uri=subject_uri)
        else:
            extra_triples = self._enrich(uri, subject_uri)

        return LinkerResult(uri=uri, extra_triples=extra_triples)

    def _enrich(self, osm_url: str, subject_uri: Optional[str]) -> str:
        """Look up an OSM feature by URL and turn it into geometry/address triples."""
        if "openstreetmap.org" not in osm_url:
            return ""
        try:
            parts = osm_url.rstrip("/").split("/")
            osm_type, osm_id = parts[-2], parts[-1]
            lookup_result = self._geocoder.lookup_osm(osm_type, osm_id)
            if lookup_result:
                extracted = self._parser.detect_and_extract(lookup_result)
                return self._parser.format_triples(extracted, subject_uri=subject_uri)
        except Exception as e:
            logger.error(f"Failed to enrich Nominatim result for {osm_url}: {e}")
        return ""
