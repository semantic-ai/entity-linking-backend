from src.linkers.base import EntityLinker, LinkerResult
from src.linkers.organization import OrganizationLinker
from src.linkers.location import LocationLinker


_org = OrganizationLinker()
_loc = LocationLinker()

LINKERS: dict[str, EntityLinker] = {
    "administrative_body": _org,
    "administrative body": _org,
    "http://www.w3.org/ns/org#organization": _org,
    "location": _loc,
    "http://purl.org/dc/terms/location": _loc,
}
