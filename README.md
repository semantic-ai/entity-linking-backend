# Entity Linking Backend

This repository contains the backend service for Named Entity Linking. It resolves recognized named entities (from an upstream NER step) to URIs using deterministic linkers — Nominatim for locations and an Elasticsearch-backed search for organizations.

## Project Structure

```
entity-linking-backend/
├── config/             # Configuration files (migrations, virtuoso)
├── data/               # Data files
├── src/
│   ├── api.py          # FastAPI entry point (health, delta endpoint)
│   ├── config.py       # Service configuration
│   ├── job.py          # Task queue orchestration
│   ├── task.py         # Named Entity Linking task processing
│   ├── nel_annotation.py # RDF annotation builder
│   ├── linkers/        # Entity linkers
│   │   ├── base.py     # Abstract base class
│   │   ├── location.py # Nominatim-based location linker
│   │   └── organization.py # Elasticsearch-based organization linker
│   ├── tools/
│   │   └── nominatim_search.py # Nominatim HTTP client
│   └── utils/
│       └── nominatim_parser.py # Nominatim result → RDF triples
└── requirements.txt
```

## Features

- **Location Linking**: Deterministic resolution via Nominatim geocoding with configurable overrides for pilot cities.
- **Organization Linking**: Resolution via Elasticsearch vector search on organization names filtered by governing unit.
- **Task Processing**: Delta-driven task queue that processes NER annotations and produces NEL annotations with provenance.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
## Installation

1.  Clone the repository.
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application is configured via environment variables and a `config.json` file.

### Environment Variables

You can set these directly or via a `.env` file:

```env
# Services
NOMINATIM_ENDPOINT=http://localhost:8080/
SEARCH_ENDPOINT=http://search
MU_SPARQL_ENDPOINT=http://virtuoso:8890/sparql

# Retry configuration
MAX_RETRIES=3
```

### External Config File (`config.json`)

The `config.json` file is used for `location_overrides` — static mappings that bypass Nominatim for known pilot-city entities. Use `config_example.json` as a template.

### Docker Volumes for Configuration

1.  **External Config**: Create a directory (e.g., `config/entitylinking`) and place your `config.json` file inside it.
2.  **Run with Docker**: Mount the config directory to `/config`.

In your `docker-compose.yml`, you can add:

```yaml
    volumes:
      - ./config/entitylinking:/config
```

## Usage

### Running Locally

```bash
python -m uvicorn web:app --host 0.0.0.0 --port 80
```
 
## Docker Compose Example

A minimal `docker-compose.yml` for running the service with Nominatim:

```yaml
services:
    nel-service:
        build: .
        volumes:
            - ./:/app
            - ./config.json:/config/config.json:ro
        ports:
            - "80:80"
        environment:
            - NOMINATIM_ENDPOINT=http://nominatim:8080/
            - MU_SPARQL_ENDPOINT=http://virtuoso:8890/sparql
        depends_on:
            - nominatim

    nominatim:
        image: mediagis/nominatim:4.2
        environment:
            - PBF_URL=https://download.geofabrik.de/europe/belgium-latest.osm.pbf
        shm_size: '1gb'
        ports:
            - "8080:8080"
```

Start the stack with:

```bash
docker compose up 
```

### Nominatim setup for multiple locations

In order to setup nominatim so it supports multiple regions a custom entry script is needed.

Example config for this can be found below:

```yaml
nominatim:
  image: mediagis/nominatim:4.2
  environment:
    - PBF_PATH=/data/merged.osm.pbf
  shm_size: '1gb'
  volumes:
    - nominatim_pbf:/data
    - nominatim_data:/var/lib/postgresql/14/main
    - ./init-nominatim.sh:/app/init-nominatim.sh
  entrypoint: /bin/bash /app/init-nominatim.sh
  ports:
    - "8080:8080"
```

How this works:

1. `PBF_PATH=/data/merged.osm.pbf` tells Nominatim to import from a single merged file.
2. `init-nominatim.sh` runs as the container entrypoint and prepares that merged file.
3. The `/data` volume (`nominatim_pbf`) persists the merged file, so repeated starts do not re-download/re-merge.
4. The PostgreSQL volume (`nominatim_data`) persists Nominatim's database.

What `init-nominatim.sh` does:

1. Checks if `/data/merged.osm.pbf` exists.
2. If missing, installs required tools (`wget`, `osmium-tool`, `ca-certificates`).
3. Downloads multiple regional extracts (currently Belgium, Oberfranken, Freiburg) from Geofabrik.
4. Merges them with `osmium merge` into `/data/merged.osm.pbf`.
5. Removes temporary per-region files to save disk space.
6. Executes the original Nominatim startup command (`/app/start.sh`) so normal import/startup continues.

To use other regions, edit `init-nominatim.sh` and replace/add `wget` input files plus the `osmium merge` input list.

### Location overrides for pilot cities

The `location` entity class (`dct:terms/Location`) is resolved deterministically via
`src/linkers/location.py`, which calls Nominatim's free-text `/search` with the
recognized entity label. On a merged multi-region OSM extract, that search can rank
the wrong point of interest for an ambiguous label, e.g. a `Rathaus` search can just
as easily surface a bus stop, a parking lot, or a town hall in an unrelated town named
the same as the target street, instead of the city hall that is wanted.

To fix a specific recurring mismatch, add an entry to `location_overrides` in
`config.json`, keyed by the entity label exactly as it's recognized in the text
(case-insensitive):

```json
{
  "location_overrides": {
    "bamberg": {
      "uri": "https://www.openstreetmap.org/way/27009786",
      "label": "Altes Rathaus, Bamberg",
      "aliases": ["Rathaus Bamberg", "Stadt Bamberg"]
    },
    "freiburg": {
      "uri": "https://www.openstreetmap.org/relation/6824",
      "lat": 47.9961443,
      "lon": 7.8491682,
      "label": "Neues Rathaus, Freiburg im Breisgau"
    }
  }
}
```

When a recognized label matches a key **or one of its `aliases`** (case-insensitive),
Nominatim search is skipped entirely and:

- **`uri` only** : the linker looks up that OSM feature directly (via `/lookup`) and
  enriches the result with its address/geometry, same as a normal search hit.
- **`uri` + `lat`/`lon`** : the linker skips the network lookup too and builds the
  location straight from the given coordinates. Use this when you already know the
  exact point and don't want a dependency on the OSM feature still existing/matching.

Use `aliases` when multiple surface forms should resolve to the same point (e.g. NER
might recognize a document as mentioning "Bamberg", "Rathaus Bamberg", or "Stadt
Bamberg" for what is really the same city hall), list the variants once instead of
duplicating the whole override block per label. `label` is only used as the output
`rdfs:label` on the linked location; it's not part of matching, so all aliases produce
the same canonical label.

`uri` is always required, it becomes the `skos:exactMatch` target for the linked
entity, so it should be a stable URI (typically the OSM node/way/relation URL, found
via a search on [nominatim.openstreetmap.org](https://nominatim.openstreetmap.org) or
[openstreetmap.org](https://www.openstreetmap.org)).

## API Endpoints

- **Health check — `GET /`**

    ```bash
    curl -s http://localhost/ | jq
    ```

    ```json
    {
        "status": "running"
    }
    ```

- **Delta notification — `POST /delta`**

    Triggers processing of open named entity linking tasks. Called by the mu-delta-notifier when new tasks appear.

    ```bash
    curl -X POST http://localhost/delta \
      -H "Content-Type: application/json" \
      -d '[{"inserts": [...], "deletes": []}]'
    ```

## Run with tasks

In a previous step of the pipeline, the NER service will have detected ELI-related entities, such as mandatees, governmental bodies...
This will used as input container for the entity linking task.

### Create NEL task with governmental body as input container

Open your local SPARQL query editor (by default configured to run on http://localhost:8890/sparql as set by lblod/app-decide), and run the following query to create a Task:
```
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>
PREFIX skolem: <http://data.lblod.info/id/.well-known/genid/>
PREFIX org: <http://www.w3.org/ns/org#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX oa: <http://www.w3.org/ns/oa#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

INSERT DATA {
  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/tasks/demo-entity-linking>
      a task:Task ;
      mu:uuid "demo-named-entity-linking" ;
      adms:status <http://redpencil.data.gift/id/concept/JobStatus/scheduled> ;
      task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/named-entity-linking> ;
      task:inputContainer <http://data.lblod.info/id/data-container/demo-entity-linking> ;
      dct:created "2025-10-31T09:00:00Z"^^xsd:dateTime .

    <http://data.lblod.info/id/data-container/demo-entity-linking> a nfo:DataContainer ;
        mu:uuid "f444f89b-78d9-497d-bd77-965923e9f864" ;
        task:hasResource <http://data.lblod.info/id/annotation/3472c89c-6869-4e04-bdb3-41a46961e9ee> .

    <http://data.lblod.info/id/annotation/3472c89c-6869-4e04-bdb3-41a46961e9ee> a oa:Annotation ;
                oa:hasBody skolem:demo-entity-linking-statement ;
                oa:hasTarget <http://data.lblod.info/id/expressions/demo-entity-linking> .

    skolem:demo-entity-linking-statement a rdf:Statement ; 
        rdf:subject <http://data.lblod.info/id/works/demo-entity-linking> ;
        rdf:predicate eli:passed_by ;
        rdf:object skolem:demo-entity-linking-administrative-body .

    skolem:demo-entity-linking-administrative-body a <http://data.vlaanderen.be/ns/besluit#Bestuursorgaan> ;
                                            rdfs:label "Vast Bureau" ;
                                            dct:spatial "Gent" .
  }
}
```

Trigger this task using
```
curl -X POST http://localhost:8080/delta \
  -H "Content-Type: application/json" \
  -d '[
    {
      "inserts": [
        {
          "subject": { "type": "uri", "value": "http://data.lblod.info/id/tasks/demo-entity-linking" },
          "predicate": { "type": "uri", "value": "http://www.w3.org/ns/adms#status" },
          "object": { "type": "uri", "value": "http://redpencil.data.gift/id/concept/JobStatus/scheduled" },
          "graph": { "type": "uri", "value": "http://mu.semte.ch/graphs/harvesting" }
        }
      ],
      "deletes": []
    }
  ]'
```

Or restart the service to pick up open tasks.

This should result in a result container added to the task:

```
<http://data.lblod.info/id/tasks/demo-entity-linking> task:resultsContainer <http://data.lblod.info/id/data-container/c703fbe0-c27c-402b-b0cb-f2ab09f6fc10> .

<http://data.lblod.info/id/data-container/c703fbe0-c27c-402b-b0cb-f2ab09f6fc10>
	rdf:type	<http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#DataContainer> ;
	<http://mu.semte.ch/vocabularies/core/uuid>	"c703fbe0-c27c-402b-b0cb-f2ab09f6fc10" ;
	<http://redpencil.data.gift/vocabularies/tasks/hasResource>	<http://data.lblod.info/id/annotations/70dcb9c5-ec0f-47d1-a87e-27c9fbba24e5> .

<http://data.lblod.info/id/annotations/70dcb9c5-ec0f-47d1-a87e-27c9fbba24e5>
	rdf:type	oa:Annotation ;
	<http://mu.semte.ch/vocabularies/core/uuid>	"70dcb9c5-ec0f-47d1-a87e-27c9fbba24e5" ;
	oa:hasTarget	<http://data.lblod.info/id/expressions/demo-entity-linking> ;
	oa:hasBody	<http://data.lblod.info/id/.well-known/genid/demo-entity-linking-statement> .

<http://data.lblod.info/id/.well-known/genid/demo-entity-linking-statement>
	rdf:type	rdf:Statement ;
	rdf:object	<http://data.lblod.info/id/.well-known/genid/demo-entity-linking-administrative-body> ;
	rdf:predicate	<http://data.europa.eu/eli/ontology#passed_by> ;
	rdf:subject	<http://data.lblod.info/id/works/demo-entity-linking> .

<http://data.lblod.info/id/.well-known/genid/demo-entity-linking-administrative-body>
	rdf:type	<http://data.vlaanderen.be/ns/besluit#Bestuursorgaan> ;
	rdfs:label	"Vast Bureau" ;
	dcterms:spatial	"Gent" ;
	skos:exactMatch	<http://data.lblod.info/id/bestuursorganen/1ab898407eb44f212df82fa0293d7e67ff2fc6c866e45b5a42e6317d27e> .
```