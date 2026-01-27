import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)


class NominatimGeocoder:
    """ Nominatim geocoder helper (lightweight)."""

    def __init__(self, base_url: str = "http://localhost:8080", rate_limit: float = 1.0, timeout: float = 10.0):
        self.base_url = base_url.rstrip('/')
        self.rate_limit = max(0.0, rate_limit)
        self.timeout = timeout
        self._last = 0.0

    async def _throttle(self) -> None:
        """Simple rate limiter based on minimum seconds between calls."""
        now = time.monotonic()
        wait = self.rate_limit - (now - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()

    async def search(self, query: str, city: Optional[str] = "Gent", country: Optional[str] = "BE", limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Query /search on the Nominatim server.
        Returned dict contains: query, display_name, lat, lon, importance, place_id,
        osm_type, osm_id, address (sub-dict), bbox, type, class, osm_url.
        """
        if not query or not query.strip():
            return None

        await self._throttle()

        # Build query string
        parts = [query.strip()]
        if city and city.strip():
            parts.append(city.strip())
        full_query = ", ".join(parts)

        params = {
            "q": full_query,
            "format": "json",
            "limit": limit,
            "addressdetails": 1,
            "extratags": 0,
            "namedetails": 0,
        }

        if country and country.strip():
            params["countrycodes"] = country.strip()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/search", params=params, timeout=self.timeout)
                resp.raise_for_status()
                results = resp.json()
                if not results:
                    return None
                return self._format(results[0], original_query=query)
        except httpx.RequestError as exc:
            logger.warning("Nominatim request failed for %r: %s", query, exc)
            return None
        except ValueError as exc:
            logger.warning("Failed parsing Nominatim JSON for %r: %s", query, exc)
            return None

    def _format(self, r: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Return compact result structure (keep only useful keys)."""
        addr = r.get("address", {})
        osm_type = r.get("osm_type")
        osm_id = r.get("osm_id")
        osm_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_type and osm_id else None

        return {
            "query": original_query,
            "display_name": r.get("display_name"),
            "lat": float(r.get("lat", 0.0)),
            "lon": float(r.get("lon", 0.0)),
            "importance": r.get("importance"),
            "place_id": r.get("place_id"),
            "osm_type": osm_type,
            "osm_id": osm_id,
            "osm_url": osm_url,
            "address": {
                "house_number": addr.get("house_number"),
                "road": addr.get("road"),
                "city": addr.get("city") or addr.get("town") or addr.get("village"),
                "postcode": addr.get("postcode"),
                "country": addr.get("country"),
                "country_code": addr.get("country_code"),
            },
            "bbox": r.get("boundingbox"),
            "type": r.get("type"),
            "class": r.get("class"),
        }
