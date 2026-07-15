# Entity Linking Backend MCP Server

Backend service for Named Entity Linking (NEL) against Linked Data sources. Exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server alongside HTTP agent endpoints that use LLM-driven tool calling to resolve entities (administrative bodies, locations, mandataries) to their canonical URIs.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Research Modes](#research-modes)
- [MCP Server](#mcp-server)
- [Task Processing (Delta Pipeline)](#task-processing-delta-pipeline)
- [Docker Deployment](#docker-deployment)
- [Nominatim Setup](#nominatim-setup-for-multiple-regions)

---

## Features

- **SPARQL Integration** — Generate and execute SPARQL queries against configured endpoints with automatic validation and prefix fixing.
- **Knowledge Base** — Semantic search over SPARQL examples, SHACL shapes, and VoID descriptions using Qdrant, in-memory embeddings, or Ollama.
- **Location Search** — Geocoding via Nominatim (self-hosted or public).
- **Web Search** — DuckDuckGo integration for general knowledge lookup.
- **Multiple LLM Providers** — OpenAI, Mistral, and Ollama (local).
- **Research Mode** — Plan-then-execute LangGraph architecture with streaming support.
- **Task Pipeline** — Delta-driven task processing compatible with the mu-semtech stack.

---

## Project Structure

```
entity-linking-backend/
├── src/
│   ├── api.py                  # FastAPI application & HTTP endpoints
│   ├── agent.py                # Agent class — orchestrates LLM + tools
│   ├── mcp_server.py           # MCP Server (FastMCP) tool definitions
│   ├── config.py               # Settings, endpoint & entity class config
│   ├── knowledge_base.py       # Vector store (Qdrant / in-memory)
│   ├── task.py                 # Named Entity Linking task processor
│   ├── job.py                  # Job/task lifecycle management
│   ├── nel_annotation.py       # Annotation reading/writing
│   ├── agent_helpers/
│   │   ├── models.py           # AgentConfig, SparqlResponse, ResearchResponse
│   │   ├── research_graph.py   # LangGraph plan-execute state machine
│   │   ├── prompts.py          # System prompts for all modes
│   │   ├── mcp_tools.py        # MCP → LangChain tool bridge
│   │   ├── serialization.py    # Message tracing & serialization
│   │   └── logging_callbacks.py
│   ├── tools/
│   │   ├── sparql_search.py    # SPARQL client
│   │   ├── nominatim_search.py # Nominatim geocoder
│   │   └── web_search.py       # DuckDuckGo search
│   ├── linkers/                # Entity-class-specific linking logic
│   └── utils/
├── config/                     # SPARQL migrations & Virtuoso config
├── data/                       # VoID files, SPARQL examples, SHACL shapes
├── docs/                       # Architecture documentation
├── config_example.json         # Template for external config
├── docker-compose.local.yml    # Local dev stack
├── Dockerfile
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerised deployment)
- An LLM provider: OpenAI API key, Mistral API key, or a running Ollama instance

---

## Installation

```bash
git clone <repository-url>
cd entity-linking-backend
pip install -r requirements.txt
```

---

## Configuration

Settings are resolved with the following priority (highest wins):

1. **Environment variables** (or `.env` file)
2. **Config file** (`config.json` mounted at `/config/config.json`)
3. **Default values** (defined in `src/config.py`)

### Environment Variables

```env
# --- LLM Provider ---
LLM_PROVIDER=openai                  # openai | mistral | ollama
LLM_MAX_RETRIES=3
LLM_REQUEST_TIMEOUT=300              # seconds
TEMPERATURE=0.0


# --- Mistral ---
MISTRAL_API_KEY=...
MISTRAL_MODEL=ministral-14b-2512
MISTRAL_ENDPOINT=                    # optional

# --- Ollama (local) ---
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral-nemo

# --- Vector Store & Embeddings ---
VECTOR_STORE_TYPE=memory_embedding   # memory_embedding | simple | qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_PROVIDER=ollama            # ollama | fastembed
EMBEDDING_DIMENSIONS=768
FORCE_INDEX=false                    # rebuild index on startup
AUTO_INIT=true                       # auto-initialise knowledge base

# --- Services ---
NOMINATIM_ENDPOINT=http://localhost:8080/
MU_SPARQL_ENDPOINT=http://virtuoso:8890/sparql

# --- MCP ---
MCP_SERVER_URL=http://localhost:80/mcp/sse

# --- Tool Selection ---
ENABLED_TOOLS=search_sparql_docs,execute_sparql_query,search_location

# --- Research Mode ---
RESEARCH_TIMEOUT=600                 # hard timeout for research requests (seconds)
RESEARCH_RECURSION_LIMIT=30          # max ReAct iterations (flat mode)

# --- Logging ---
VERBOSE=false
TRACING_ENABLED=false
```

### Config File (`config.json`)

Use `config_example.json` as a template. The config file defines SPARQL endpoints and per-entity-class tool/template mappings.

```json
{
  "endpoints": [
    {
      "endpoint_url": "http://virtuoso:8890/sparql",
      "void_file": "data/queries/local/local_sparql_void.ttl",
      "examples_file": "data/queries/local/local_sparql_examples.ttl"
    }
  ]
}
```

When a structured request arrives, the backend JIT-spawns a lightweight agent scoped to only the tools mapped for that entity class. This is especially useful for smaller models that struggle with too many tools.

### LLM Providers

| Provider | When to use | Config |
|----------|-------------|--------|
| **OpenAI** | Best overall performance | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` |
| **Mistral** | Good balance of cost/performance | `LLM_PROVIDER=mistral` + `MISTRAL_API_KEY` |
| **Ollama** | Local/private, no API costs | `LLM_PROVIDER=ollama` + `OLLAMA_HOST` |

Tested local models:
- **Mistral Nemo** — Decent performance for moderate queries, functional tool-calling.
- **Ministral 14b** — Best results via Mistral API; tool-calling issues when served through Ollama.

---

## API Endpoints

The service runs on port **80** by default. All endpoints are JSON-based.

### `GET /` — Health Check

```bash
curl -s http://localhost/ | jq
```

```json
{ "status": "running", "endpoints": ["/mcp"] }
```

---

### `POST /agent/query` — Free-form Entity Linking

Run a general-purpose entity linking query. The agent uses all enabled tools and returns structured results.

**Request:**

```json
{ "query": "Find the OpenStreetMap URI for Station Gent-Sint-Pieters" }
```

**Response:** `SparqlResponse`

```json
{
  "results": [
    {
      "uri": "https://www.openstreetmap.org/node/123456",
      "label": "Station Gent-Sint-Pieters",
      "location": "Gent",
      "reasoning": "Matched by name in Nominatim search..."
    }
  ]
}
```

---

### `POST /agent/query_structured` — Structured Entity Linking

Target a specific entity class. The agent is scoped to only the tools and template defined for that class in `config.json`.

**Request:**

```json
{
  "entity_class": "Administrative Body",
  "entity_label": "Vast Bureau",
  "location": "Gent"
}
```

**Response:** `SparqlResponse` (same schema as above)

Supported entity classes (configurable via `config.json`):
- **Administrative Body** — resolves via SPARQL
- **Location** — resolves via Nominatim
- **Mandatary** — resolves via SPARQL (Flemish municipalities only; requires "centrale vindplaats" endpoint)

---

### `POST /agent/research` — Research Query

Free-form research with full message trace and tool-call breakdown. Supports conversation history via the `messages` field.

**Request:**

```json
{
  "query": "Which SPARQL endpoint and query pattern should I use to find administrative bodies in Gent?",
  "messages": [
    { "role": "user", "content": "previous question..." },
    { "role": "assistant", "content": "previous answer..." }
  ]
}
```

**Response:** `ResearchResponse`

```json
{
  "answer": "Based on the available endpoints...",
  "sources": ["http://..."],
  "sparql_results": [{ "tool": "execute_sparql_query", "result": "..." }],
  "messages": [...],
  "tool_calls": [...],
  "tool_results": [...],
  "trace": "Human-readable execution trace",
  "raw_response": { "plan": [...], "step_results": [...] }
}
```

---

### `POST /agent/research/stream` — Streaming Research (SSE)

Same as `/agent/research` but streams progress as Server-Sent Events. Useful for real-time UI updates.

**Request:** Same as `/agent/research`

**Response:** `text/event-stream` with JSON events:

| Event | Description |
|-------|-------------|
| `graph` | Graph structure (Mermaid diagram) |
| `retrieve` | Retrieval phase completed |
| `plan` | Initial plan generated |
| `step_done` | A plan step completed |
| `replan` | Plan revised after failure |
| `validate` | Final answer synthesized |
| `done` | Execution complete |
| `error` | An error occurred |

**Example client:**

```javascript
const response = await fetch('/agent/research/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: "Find all municipalities in West-Flanders" })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      console.log(`[${event.event}]`, event.data);
    }
  }
}
```

---

### `POST /delta` — Delta Notification (Task Trigger)

Receives delta notifications from the mu-semtech stack to trigger task processing.

```bash
curl -X POST http://localhost:80/delta \
  -H "Content-Type: application/json" \
  -d '[{"inserts": [...], "deletes": []}]'
```

Returns `202 Accepted` and processes tasks in the background.

---

## Research Mode

The research endpoint supports two execution strategies, controlled by the `planning_enabled` setting (default: `true`).

### Flat ReAct Mode (`planning_enabled: false`)

A single ReAct agent loop:

1. LLM reads the research system prompt (5-step methodology: analyse → retrieve docs → construct queries → evaluate → synthesize).
2. Iteratively calls tools (`search_sparql_docs`, `execute_sparql_query`, etc.).
3. Terminates when a final answer is produced or `research_recursion_limit` is reached.
4. Hard timeout via `research_timeout`.

Best for: Simple queries, lower latency, smaller models.

### Plan-Execute Graph Mode (`planning_enabled: true`)

A LangGraph state machine with structured planning:

```
START → retrieve → plan → execute ⇄ monitor → validate → END
                              ↑         ↓
                              └── replan ┘
```

| Node | Purpose |
|------|---------|
| **retrieve** | Gathers relevant SPARQL documentation, schemas, and examples |
| **plan** | LLM generates a step-by-step plan (JSON array of `PlanStep` objects) |
| **execute** | Runs a ReAct agent scoped to each step's tools and goal |
| **monitor** | Pure Python — detects stuck loops, timeouts, empty results |
| **replan** | LLM revises remaining steps based on what worked/failed |
| **validate** | LLM synthesizes the final answer from all step results |

**Monitoring heuristics:**
- Consecutive empty results → trigger replan
- Same tool called repeatedly → force next step
- Per-step tool call cap (`max_step_tool_calls`) → move on
- Per-step timeout (`step_timeout_s`) → skip or replan

Best for: Complex multi-hop queries, higher accuracy, detailed audit trail.

### Research Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `research_timeout` | 600s | Hard timeout for the entire research request |
| `research_recursion_limit` | 30 | Max ReAct iterations (flat mode) |
| `planning_enabled` | `true` | Use plan-execute graph vs flat ReAct |
| `streaming_enabled` | `true` | Enable SSE streaming for research |
| `step_timeout_s` | 90s | Per-step timeout in plan-execute mode |
| `max_step_tool_calls` | 6 | Max tool invocations per plan step |
| `max_replans` | 2 | Max plan revisions |
| `max_interventions_per_step` | 2 | Max monitor nudges before skipping a step |

---

## MCP Server

The MCP server is mounted at `/mcp` and exposes the following tools to any MCP-compatible client:

| Tool | Description |
|------|-------------|
| `search_sparql_docs` | Retrieves relevant SPARQL examples and class schemas from the knowledge base |
| `execute_sparql_query` | Executes a SPARQL query with automatic validation and prefix fixing |
| `search_location` | Geocodes a location query via Nominatim |
| `search_web` | Web search via DuckDuckGo |

Connect an MCP client (e.g., Claude Desktop, VS Code Copilot, or any `fastmcp`-compatible client):

```bash
# SSE transport
curl -N -H "Accept: text/event-stream" http://localhost/mcp/sse
```

Tool availability is controlled by the `ENABLED_TOOLS` environment variable.

---

## Task Processing (Delta Pipeline)

The service integrates with the [mu-semtech](https://mu.semte.ch/) microservice stack for automated entity linking as part of a larger pipeline.

### How It Works

1. A Named Entity Recognition (NER) service detects entities in documents and creates `oa:Annotation` resources.
2. A `task:Task` with operation `named-entity-linking` is created in the triplestore.
3. The service picks up scheduled tasks (via `/delta` notification or on startup).
4. For each annotation, the agent resolves the entity to a URI and writes the result back as `skos:exactMatch`.

### Creating a Demo Task

Insert the task into your SPARQL endpoint (default: `http://localhost:8890/sparql`):

```sparql
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>
PREFIX skolem: <http://data.lblod.info/id/.well-known/genid/>
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

### Triggering the Task

```bash
curl -X POST http://localhost:80/delta \
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

Alternatively, restart the service — it drains open tasks on startup.

### Expected Result

The task's result container will contain the original annotation enriched with `skos:exactMatch`:

```turtle
<http://data.lblod.info/id/.well-known/genid/demo-entity-linking-administrative-body>
    a <http://data.vlaanderen.be/ns/besluit#Bestuursorgaan> ;
    rdfs:label "Vast Bureau" ;
    dct:spatial "Gent" ;
    skos:exactMatch <http://data.lblod.info/id/bestuursorganen/...> .
```

---

## Docker Deployment

### Local Development Stack

```bash
docker compose -f docker-compose.local.yml up -d
```

This starts:
- **decide-mcp** — The entity linking service (port 80)
- **qdrant** — Vector database (port 6333)
- **ollama** — Local LLM serving (port 11434, auto-pulls `mistral-nemo`)
- **virtuoso** — SPARQL triplestore (port 8890)
- **migrations** — Runs SPARQL migrations on startup

### Production Deployment

```yaml
services:
  decide-mcp:
    build: .
    ports:
      - "80:80"
    env_file:
      - .env
    environment:
      - LLM_PROVIDER=openai
      - QDRANT_HOST=qdrant
      - MU_SPARQL_ENDPOINT=http://virtuoso:8890/sparql
      - MCP_SERVER_URL=http://localhost:80/mcp/sse
      - ENABLED_TOOLS=search_sparql_docs,execute_sparql_query,search_location
    volumes:
      - ./config.json:/config/config.json
      - ./data:/app/data
    depends_on:
      - qdrant
      - virtuoso

  qdrant:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage

  virtuoso:
    image: redpencil/virtuoso:1.4.0-rc.1
    environment:
      SPARQL_UPDATE: 'true'
    volumes:
      - ./config/virtuoso/virtuoso.ini:/data/virtuoso.ini
      - virtuoso_data:/data

volumes:
  qdrant_data:
  virtuoso_data:
```

### Running Without Docker

```bash
# Set environment variables or create .env file
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Start the server
python -m src.api
```

The server starts on `http://0.0.0.0:80`.

---

## Nominatim Setup for Multiple Regions

To support geocoding across multiple regions, use a custom entrypoint script that merges OSM extracts:

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

**How `init-nominatim.sh` works:**

1. Checks if `/data/merged.osm.pbf` exists.
2. If missing: installs `wget`, `osmium-tool`; downloads regional extracts from Geofabrik (Belgium, Oberfranken, Freiburg by default).
3. Merges them with `osmium merge` into a single PBF file.
4. Cleans up temporary files.
5. Starts Nominatim import via `/app/start.sh`.

To add/change regions, edit the download URLs in `init-nominatim.sh`.
