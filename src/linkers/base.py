from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class LinkerResult(BaseModel):
    uri: str
    extra_triples: str = ""


class EntityLinker(ABC):
    @abstractmethod
    def link(
        self,
        entity_label: str,
        location: str,
        subject_uri: Optional[str] = None,
    ) -> Optional[LinkerResult]:
        """Resolve a label (+ optional location filter) to a URI.

        Returns None when no confident match exists; callers should skip
        annotation creation in that case.

        Args:
            entity_label: The label of the entity to resolve.
            location: A region/location string used as a filter or geographic
                hint. May be the literal "Unknown location" sentinel from
                upstream; implementations decide how to handle that.
            subject_uri: The URI of the source-document entity being linked.
                Implementations that decorate the subject with auxiliary
                triples (e.g. dcterms:Location, geometry) use this as the
                triple subject. Ignored by implementations that only return
                a plain URI match.
        """
        ...
