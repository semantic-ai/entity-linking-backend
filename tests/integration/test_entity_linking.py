import re

from . import tasks

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def test_an_overridden_location_links_to_the_configured_uri(search_services):
    task_uri = "http://example.org/test/task/link-override"
    entity_uri = "http://example.org/test/entity/link-override"
    tasks.seed("link_override")
    requests_before = len(search_services)

    tasks.run(task_uri)

    assert tasks.status(task_uri) == tasks.SUCCESS
    assert tasks.links("http://example.org/test/target/link-override") == [
        "https://www.openstreetmap.org/relation/6824"
    ]
    # label from config.json's location_overrides, point geometry from its lat/lon
    assert (RDFS_LABEL, "Neues Rathaus, Freiburg im Breisgau") in tasks.entity(entity_uri)
    geometry = tasks.wkt(entity_uri)
    # Virtuoso rewrites the literal and swaps the axes of EPSG:4326 coordinates, so compare values only.
    assert geometry.startswith("POINT")
    assert set(re.findall(r"-?\d+\.\d+", geometry)) == {"7.8491682", "47.9961443"}
    assert search_services[requests_before:] == []
