from . import tasks

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


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
    assert tasks.wkt(entity_uri).startswith("POINT")
    assert tasks.coordinates(entity_uri) == {7.8491682, 47.9961443}
    assert search_services[requests_before:] == []


def test_a_location_is_resolved_through_nominatim_within_the_governing_unit(search_services):
    task_uri = "http://example.org/test/task/link-nominatim"
    entity_uri = "http://example.org/test/entity/link-nominatim"
    tasks.seed("link_nominatim")
    requests_before = len(search_services)

    tasks.run(task_uri)

    assert tasks.status(task_uri) == tasks.SUCCESS
    assert [q["q"] for path, q in search_services[requests_before:] if path == "/search"] == [["Kerkstraat, Gent"]]
    assert tasks.links("http://example.org/test/target/link-nominatim") == [
        "https://www.openstreetmap.org/way/123"
    ]
    assert (RDF_TYPE, "https://data.vlaanderen.be/ns/adres#Straatnaam") in tasks.entity(entity_uri)
    assert tasks.wkt(entity_uri).startswith("LINESTRING")
    assert tasks.coordinates(entity_uri) == {3.72, 3.73, 51.05, 51.06}


def test_an_organization_is_resolved_through_search_within_the_governing_unit(search_services):
    task_uri = "http://example.org/test/task/link-organization"
    target_uri = "http://example.org/test/target/link-organization"
    tasks.seed("link_organization")
    requests_before = len(search_services)

    tasks.run(task_uri)

    assert tasks.status(task_uri) == tasks.SUCCESS
    assert [
        (q["filter[name.*]"], q["filter[:term:owning-body]"])
        for path, q in search_services[requests_before:]
        if path == "/organizations/search/"
    ] == [(["OCMW"], ["http://example.org/test/governing-unit/gent"])]
    assert tasks.links(target_uri) == ["http://example.org/test/organization/ocmw-gent"]
    assert tasks.search_scores(target_uri) == [12.5]


def test_linking_the_same_entity_twice_writes_one_annotation(search_services):
    tasks.seed("link_twice")

    for task_uri in (
        "http://example.org/test/task/link-twice-1",
        "http://example.org/test/task/link-twice-2",
    ):
        tasks.run(task_uri)
        assert tasks.status(task_uri) == tasks.SUCCESS

    assert tasks.links("http://example.org/test/target/link-twice") == [
        "http://example.org/test/organization/ocmw-gent"
    ]
