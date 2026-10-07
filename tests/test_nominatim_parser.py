"""Unit tests for ``src/utils/nominatim_parser.py``."""
import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from src.utils.nominatim_parser import NominatimParser

DCT = Namespace("http://purl.org/dc/terms/")
LOCN = Namespace("http://www.w3.org/ns/locn#")
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

OSM_WAY = "https://www.openstreetmap.org/way/123"
WGS84 = "<http://www.opengis.net/def/crs/EPSG/0/4326>"


@pytest.fixture
def parser():
    return NominatimParser()


def address_answer(**overrides):
    answer = {
        "osm_type": "way",
        "osm_id": 123,
        "display_name": "25, Kerkstraat, Gent, Oost-Vlaanderen, Belgium",
        "lat": "51.05",
        "lon": "3.72",
        "class": "building",
        "type": "house",
        "address": {
            "house_number": "25",
            "road": "Kerkstraat",
            "city": "Gent",
            "postcode": "9000",
        },
    }
    answer.update(overrides)
    return answer


class TestDetectAndExtract:
    def test_an_address_with_a_house_number_gets_a_composed_label(self, parser):
        info = parser.detect_and_extract(address_answer())

        assert info["type"] == "Address"
        assert info["label"] == "Kerkstraat 25, 9000 Gent"
        assert info["street"] == "Kerkstraat"
        assert info["num"] == "25"
        assert info["post_code"] == "9000"
        assert info["post_name"] == "Gent"

    def test_addresstags_given_as_a_list_are_read(self, parser):
        info = parser.detect_and_extract(
            {
                "osm_type": "W",
                "osm_id": 123,
                "localname": "Kerkstraat",
                "category": "building",
                "type": "house",
                "addresstags": [
                    {"type": "housenumber", "localname": "25"},
                    {"type": "postcode", "localname": "9000"},
                    {"type": "city", "localname": "Gent"},
                ],
            }
        )

        assert info["type"] == "Address"
        assert info["label"] == "Kerkstraat 25, 9000 Gent"

    def test_a_highway_is_a_street(self, parser):
        info = parser.detect_and_extract(
            {"osm_type": "way", "osm_id": 123, "class": "highway", "type": "primary", "display_name": "Kerkstraat, Gent"}
        )

        assert info["type"] == "Street"
        assert info["label"] == "Kerkstraat, Gent"

    @pytest.mark.parametrize("admin_level", [8, 9, 10])
    def test_a_boundary_at_admin_level_8_to_10_is_a_sub_municipality(self, parser, admin_level):
        info = parser.detect_and_extract(
            {"osm_type": "relation", "osm_id": 5, "class": "boundary", "type": "administrative", "admin_level": admin_level}
        )

        assert info["type"] == "Sub-municipality"

    @pytest.mark.parametrize("admin_level", [4, 7, 11])
    def test_a_boundary_at_another_admin_level_gets_no_type(self, parser, admin_level):
        info = parser.detect_and_extract(
            {"osm_type": "relation", "osm_id": 5, "class": "boundary", "type": "administrative", "admin_level": admin_level}
        )

        assert "type" not in info

    def test_a_neighbourhood_is_recognised(self, parser):
        info = parser.detect_and_extract(
            {"osm_type": "node", "osm_id": 9, "class": "place", "type": "neighbourhood"}
        )

        assert info["type"] == "Neighbourhood"

    @pytest.mark.parametrize(
        "osm_type,expected",
        [
            ("W", "https://www.openstreetmap.org/way/123"),
            ("N", "https://www.openstreetmap.org/node/123"),
            ("R", "https://www.openstreetmap.org/relation/123"),
            ("way", "https://www.openstreetmap.org/way/123"),
        ],
    )
    def test_exact_match_is_the_openstreetmap_url_of_the_feature(self, parser, osm_type, expected):
        info = parser.detect_and_extract({"osm_type": osm_type, "osm_id": 123})

        assert info["exact_match"] == expected


class TestGeojsonToWkt:
    def test_geometry_wins_over_lat_lon(self, parser):
        data = {
            "geometry": {"type": "LineString", "coordinates": [[3.72, 51.05], [3.73, 51.06]]},
            "lat": "50.0",
            "lon": "4.0",
        }

        assert parser.geojson_to_wkt(data) == "LINESTRING (3.72 51.05, 3.73 51.06)"

    def test_an_unparsable_geometry_falls_back_to_the_lat_lon_point(self, parser):
        data = {"geometry": {"type": "Nonsense", "coordinates": []}, "lat": "51.05", "lon": "3.72"}

        assert parser.geojson_to_wkt(data) == "POINT (3.72 51.05)"

    def test_lat_lon_alone_gives_a_point_with_longitude_first(self, parser):
        assert parser.geojson_to_wkt({"lat": 47.9961443, "lon": 7.8491682}) == "POINT (7.8491682 47.9961443)"

    def test_neither_geometry_nor_lat_lon_gives_none(self, parser):
        assert parser.geojson_to_wkt({"display_name": "Gent"}) is None


class TestFormatTriples:
    @staticmethod
    def parse(ntriples):
        graph = Graph()
        graph.parse(data=ntriples, format="nt")
        return graph

    def test_a_given_subject_uri_is_used_as_the_location(self, parser):
        subject = URIRef("http://example.org/test/entity/1")
        info = {"label": "Kerkstraat", "exact_match": OSM_WAY}

        graph = self.parse(parser.format_triples(info, subject_uri=str(subject)))

        assert set(graph) == {
            (subject, RDF.type, DCT.Location),
            (subject, RDFS.label, Literal("Kerkstraat")),
            (subject, SKOS.exactMatch, URIRef(OSM_WAY)),
        }

    def test_without_a_subject_uri_the_location_is_a_skolem_iri(self, parser):
        graph = self.parse(parser.format_triples({"label": "Kerkstraat"}, skolem_id="kerkstraat"))

        skolem = URIRef("http://www.example.org/id/.well-known/genid/kerkstraat")
        assert (skolem, RDF.type, DCT.Location) in graph

    def test_an_address_gets_thoroughfare_and_locator_designator(self, parser):
        subject = URIRef("http://example.org/test/entity/1")
        info = parser.detect_and_extract(address_answer())

        graph = self.parse(parser.format_triples(info, subject_uri=str(subject)))

        assert (subject, RDF.type, LOCN.Address) in graph
        assert set(graph.objects(subject, LOCN.thoroughfare)) == {Literal("Kerkstraat")}
        assert set(graph.objects(subject, LOCN.locatorDesignator)) == {Literal("25")}
        assert set(graph.objects(subject, LOCN.postCode)) == {Literal("9000")}
        assert set(graph.objects(subject, LOCN.postName)) == {Literal("Gent")}

    def test_the_geometry_is_a_wkt_literal_in_epsg_4326(self, parser):
        subject = URIRef("http://example.org/test/entity/1")
        info = {"label": "Kerkstraat", "wkt": "POINT (3.72 51.05)"}

        graph = self.parse(parser.format_triples(info, subject_uri=str(subject)))

        [geometry] = graph.objects(subject, LOCN.geometry)
        assert (geometry, RDF.type, LOCN.Geometry) in graph
        assert set(graph.objects(geometry, GEOSPARQL.asWKT)) == {
            Literal(f"{WGS84} POINT (3.72 51.05)", datatype=GEOSPARQL.wktLiteral)
        }
