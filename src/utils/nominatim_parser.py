import json
import uuid
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS
from shapely.geometry import shape, Point

# Define Namespaces
MU = Namespace("http://mu.semte.ch/vocabularies/core/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
LOCN = Namespace("http://www.w3.org/ns/locn#")
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")
SKOLEM = Namespace("http://www.example.org/id/.well-known/genid/")
VLAANDEREN_ADRES = Namespace("https://data.vlaanderen.be/ns/adres#")
SCHEMA = Namespace("http://schema.org/")
WIKIDATA = Namespace("http://www.wikidata.org/entity/")
LBLOD_GEOM = Namespace("http://data.lblod.info/id/geometries/")

class NominatimParser:
    def __init__(self):
        self.reset_graph()
        
    def reset_graph(self):
        self.g = Graph()
        self.g.bind("mu", MU)
        self.g.bind("dcterms", DCTERMS)
        self.g.bind("locn", LOCN)
        self.g.bind("geosparql", GEOSPARQL)
        self.g.bind("skolem", SKOLEM)
        self.g.bind("skos", SKOS)
        self.g.bind("schema", SCHEMA)

    def _get_osm_uri(self, osm_type, osm_id):
        """Constructs the exactMatch URI based on OSM type and ID."""
        type_map = {'W': 'way', 'N': 'node', 'R': 'relation', 'way': 'way', 'node': 'node', 'relation': 'relation'}
        t = type_map.get(osm_type, 'way')
        return f"https://www.openstreetmap.org/{t}/{osm_id}"

    def geojson_to_wkt(self, data):
        if data.get("geometry"):
            try:
                return shape(data.get("geometry")).wkt
            except: pass
        if "lon" in data and "lat" in data:
            try:
                return Point(float(data["lon"]), float(data["lat"])).wkt
            except: pass
        return None

    def detect_and_extract(self, data):
        """Detects type and extracts relevant info from Nominatim JSON."""
        category = data.get("category") or data.get("class", "")
        osm_type_val = data.get("type", "")
        
        # handle address info whether it's list, dict or under addresstags
        addr_info = data.get("addresstags") or data.get("address", {})
        if isinstance(addr_info, list):
            # sometimes address comes as a list of dicts, let's map it
            addr_tags = {item.get("type"): item.get("localname") for item in addr_info if isinstance(item, dict)}
        else:
            addr_tags = addr_info

        print("category:", category, "osm_type_val:", osm_type_val, "addr_tags:", addr_tags)

        # Handle different key names for fields
        pc = addr_tags.get("postcode") or data.get("calculated_postcode", "")
        city = addr_tags.get("city") or addr_tags.get("town") or addr_tags.get("village") or addr_tags.get("municipality", "")
        num = addr_tags.get("housenumber") or addr_tags.get("house_number") or data.get("housenumber")
        street = addr_tags.get("street") or addr_tags.get("road") or data.get("localname")
        
        label_fallback = data.get("display_name") or data.get("localname") or data.get("names", {}).get("name", "Unknown")

        extracted = {
            "wkt": self.geojson_to_wkt(data),
            "label": label_fallback,
            "exact_match": self._get_osm_uri(data.get("osm_type"), data.get("osm_id")),
            "tags": addr_tags,
            "post_code": pc or "",
            "post_name": city or ""
        }

        # 1. Address
        if num:
            extracted["type"] = "Address"
            thoroughfare = (street or "").strip()
            locator = (num or "").strip()
            postal = " ".join(part for part in [extracted["post_code"], extracted["post_name"]] if part).strip()
            address_head = " ".join(part for part in [thoroughfare, locator] if part).strip()
            extracted["label"] = ", ".join(part for part in [address_head, postal] if part)
            extracted["street"] = thoroughfare
            extracted["num"] = locator
        
        # 2. Street
        elif category == "highway" or osm_type_val in ["residential", "tertiary", "secondary"]:
            extracted["type"] = "Street"
        
        # 3. Sub-municipality / Admin
        elif category == "boundary" and data.get("admin_level") in [8, 9, 10]:
            extracted["type"] = "Sub-municipality"
            
        # 4. Neighbourhood
        elif osm_type_val in ["neighbourhood", "suburb"] or data.get("addresstype") == "neighbourhood":
            extracted["type"] = "Neighbourhood"

        return extracted

    def format_triples(self, info, skolem_id="location_uuid", subject_uri=None):
        self.reset_graph()
        if subject_uri:
            loc_node = URIRef(subject_uri)
        else:
            loc_node = SKOLEM[skolem_id]
        
        self.g.add((loc_node, RDF.type, DCTERMS.Location))
        
        if info.get("label"):
            self.g.add((loc_node, RDFS.label, Literal(info["label"])))
            
        if info.get("exact_match"):
            self.g.add((loc_node, SKOS.exactMatch, URIRef(info["exact_match"])))

        # Geometry
        if info.get("wkt"):
            geom_uuid = str(uuid.uuid4())
            geom_node = LBLOD_GEOM[geom_uuid]
            self.g.add((geom_node, RDF.type, LOCN.Geometry))
            self.g.add((geom_node, MU.uuid, Literal(geom_uuid)))
            wkt_lit = Literal(f"<http://www.opengis.net/def/crs/EPSG/0/4326> {info['wkt']}", 
                              datatype=GEOSPARQL.wktLiteral)
            if info.get("post_code"):
                self.g.add((loc_node, LOCN.postCode, Literal(info["post_code"])))
            if info.get("post_name"):
                self.g.add((loc_node, LOCN.postName, Literal(info["post_name"])))

            self.g.add((geom_node, GEOSPARQL.asWKT, wkt_lit))
            self.g.add((loc_node, LOCN.geometry, geom_node))

        # Type Specifics
        t = info.get("type")
        if t == "Address":
            self.g.add((loc_node, RDF.type, LOCN.Address))
            if info.get("street"):
                self.g.add((loc_node, LOCN.thoroughfare, Literal(info["street"])))
            if info.get("num"):
                self.g.add((loc_node, LOCN.locatorDesignator, Literal(info["num"])))
        elif t == "Street":
            self.g.add((loc_node, RDF.type, VLAANDEREN_ADRES.Straatnaam))
        elif t == "Sub-municipality":
            self.g.add((loc_node, RDF.type, WIKIDATA.Q2785216))
        elif t == "Neighbourhood":
            self.g.add((loc_node, RDF.type, WIKIDATA.Q123705))

        return self.g.serialize(format="nt")


