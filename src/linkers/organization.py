from typing import Optional

from helpers import logger

from src.linkers.base import EntityLinker, LinkerResult


class OrganizationLinker(EntityLinker):
    """Resolve an organization label to a URI via vector search.

    (TODO): Elasticsearch vector search on the org
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
        logger.warning(
            "OrganizationLinker stub invoked (no Elasticsearch backend yet): "
            f"entity_label={entity_label!r}, location={location!r}, "
            f"subject_uri={subject_uri!r}"
        )
        return None
