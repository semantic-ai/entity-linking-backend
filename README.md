# Decide Research Backend

Backend for research over linked-data endpoints and indexed local-government decisions. The service combines a FastAPI API, a LangGraph plan-execute workflow, and an MCP server exposing SPARQL and document-search tools.

The repository is now research-focused. The former named-entity-linking task pipeline, structured entity-linking endpoints, and `/delta` worker were removed in commit `bd7ae9b`. They are not part of the supported runtime.

## Status

| Capability | Status |
|---|---|
| Plan-execute SPARQL research graph | Implemented |
| JSON research endpoint | Implemented; always non-streaming |
| SSE research progress endpoint | Implemented |
| SPARQL documentation retrieval and query execution | Implemented |
| Hybrid semantic/BM25 expression search | Implemented (V3 Phase 1) |
| Automatic RAG/research/follow-up routing | Planned (V3 Phase 2) |
| `/agent/ask` JSON and `/agent/ask/stream` SSE separation | Planned (V3 Phase 2) |
| Shared URI scratchpad and `candidate_uris` filtering | Planned (V3 Phase 3) |
| SPARQL-first content-search prompting | Planned (V3 Phase 4) |

The V3 roadmap is described in [research-mode-v3-plan.md](docs/research-mode-v3-plan.md). Phase 2 will add `/agent/ask` as the canonical multimode endpoint while preserving the existing research endpoints. Streaming and non-streaming behavior will be selected exclusively by `/stream` versus non-`/stream` URLs, as recorded in the [implementation spec](docs/phase2-implementation-spec.md) and [cleanup audit](docs/research-mode-v3-cleanup-audit.md).

## Current Architecture

```text
HTTP client
    |
    +-- POST /agent/research -----------+
    |                                   |
    +-- POST /agent/research/stream ----+--> Agent --> LangGraph
    |                                                  |
    +-- /mcp/sse (external MCP clients)                |
                                                       v
                 retrieve -> plan -> execute <-> replan -> validate
                    |                    |
                    v                    v
          search_sparql_docs     execute_sparql_query
                                search_expressions
```

The graph performs these stages:

1. `retrieve` finds SPARQL examples and SHACL/VoID schema documentation.
2. `plan` creates a minimal structured plan.
3. `execute` runs one step at a time with a bounded ReAct sub-agent.
4. `monitor` routes failed or empty-looking steps through `replan`.
5. `validate` synthesizes the final grounded answer.

The graph is the only supported research engine. The former flat ReAct path and its planning/streaming switches have been removed.

## Repository Layout

```text
entity-linking-backend/
├── src/
│   ├── api.py                         # HTTP and SSE routes
│   ├── agent.py                       # LLM, MCP, and graph orchestration
│   ├── mcp_server.py                  # MCP tool definitions
│   ├── config.py                      # Environment settings and endpoint config
│   ├── knowledge_base.py              # SPARQL documentation stores
│   ├── embeddings.py                  # FastEmbed/Ollama embeddings for the docs KB
│   ├── agent_helpers/
│   │   ├── research_graph.py          # LangGraph state and nodes
│   │   ├── prompts.py                 # Research prompts
│   │   ├── mcp_tools.py               # MCP-to-LangChain adapter
│   │   └── serialization.py           # JSON-safe graph response conversion
│   └── tools/
│       ├── elastic_search.py           # Hybrid decision search and enrichment
│       ├── nominatim_search.py         # Nominatim geocoding
│       ├── sparql_search.py            # SPARQL HTTP client
│       └── web_search.py               # DuckDuckGo search
├── config/
│   ├── search/                         # mu-search index definition
│   ├── embedding/                      # embedding-service configuration
│   └── authorization/                  # optional local mu-stack policy
├── data/queries/                       # SPARQL examples, VoID, and SHACL data
├── docs/                               # V3 specifications and audit
├── tests/                              # Cleanup and research regression tests
├── development.ipynb                   # Retained development notebook
├── langgraph.ipynb                     # Retained graph notebook
├── streamlit_research_app.py           # Retained developer research UI
├── config.json                         # Active SPARQL endpoint metadata
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── web.py                              # mu-python-template entry point
```

The Nominatim `search_location` and DuckDuckGo `search_web` tools are retained as supported discovery tools. They remain available through MCP and can be included in the internal Agent allowlist when required.

## API

The service listens on port `80` in the container.

### Health

```http
GET /
```

```json
{
  "status": "running",
  "endpoints": ["/mcp/sse", "/agent/research", "/agent/research/stream"]
}
```

### Research request

```http
POST /agent/research
Content-Type: application/json
```

```json
{
  "query": "Vergelijk de besluiten over renovaties in Gent en Antwerpen.",
  "messages": [
    {"role": "user", "content": "We onderzoeken lokale renovatiebesluiten."},
    {"role": "assistant", "content": "Welke vergelijking wil je maken?"}
  ]
}
```

The response uses `ResearchResponse`:

```json
{
  "answer": "...",
  "sources": null,
  "sparql_results": null,
  "messages": null,
  "tool_calls": null,
  "tool_results": null,
  "trace": null,
  "raw_response": {
    "plan": [],
    "step_results": []
  }
}
```

This route always returns one JSON response. Streaming is available only through the dedicated endpoint below.

### Streaming research request

```http
POST /agent/research/stream
Content-Type: application/json
```

The request body is identical to `/agent/research`. The response is `text/event-stream`; each `data:` record contains an object with `event` and `data` fields.

| Event | Meaning |
|---|---|
| `graph` | Compiled Mermaid graph |
| `retrieve` | Documentation retrieval completed |
| `plan` | Initial plan created |
| `step_done` | One execution step completed |
| `replan` | Remaining plan replaced |
| `validate` | Final answer synthesized |
| `done` | Stream completed |
| `error` | Execution failed |

The old `/agent/query`, `/agent/query_structured`, and `/delta` endpoints no longer exist.

## MCP Tools

The pinned FastMCP SSE application is mounted at `/mcp/sse`. The internal client defaults to `http://localhost:80/mcp/sse`.

| Tool | Role | Support |
|---|---|---|
| `search_sparql_docs` | Retrieve SPARQL examples and endpoint schemas | Core |
| `execute_sparql_query` | Validate, repair, and execute SPARQL | Core |
| `search_expressions` | Semantic, keyword, or hybrid decision search | Core |
| `search_location` | Nominatim geocoding | Supported discovery tool |
| `search_web` | DuckDuckGo search | Supported discovery tool |

`AGENT_ENABLED_TOOLS` filters only the tools loaded by the internal Agent. Every registered tool remains available to external MCP clients.

### Hybrid expression search

`search_expressions` supports:

- semantic search: `vector=true`, without `keyword`;
- hybrid search: `vector=true`, with `keyword`;
- BM25 search: `vector=false`, with `keyword`.

It queries mu-search, then enriches matching expression URIs from the configured SPARQL endpoint. Vector search requires `EMBEDDING_API_URL`; without it, the current implementation cannot create a query embedding.

## Configuration

Scalar runtime settings come from environment variables evaluated by `src/config.py`. `config.json` does not override those scalar values; it supplies only endpoint metadata such as endpoint URLs, example files, VoID files, and SHACL folders.

Minimal environment example:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1

MCP_SERVER_URL=http://localhost:80/mcp/sse
AGENT_ENABLED_TOOLS=search_sparql_docs,execute_sparql_query,search_expressions

RESEARCH_RETRIEVE_TOOLS=search_sparql_docs
RESEARCH_STEP_TIMEOUT_S=90
RESEARCH_MAX_STEP_TOOL_CALLS=6
RESEARCH_MAX_REPLANS=2

MU_SPARQL_ENDPOINT=http://virtuoso:8890/sparql
VECTOR_STORE_TYPE=simple

SEARCH_ENDPOINT=http://search
EMBEDDING_API_URL=http://ollama:11434
EMBEDDING_MODEL=embeddinggemma
ELASTIC_SEARCH_KEYWORD_FIELDS=description.*
```

### Active setting groups

| Group | Variables |
|---|---|
| LLM | `LLM_PROVIDER`, provider API key/endpoint/model variables, `TEMPERATURE`, `LLM_MAX_RETRIES`, `LLM_REQUEST_TIMEOUT` |
| Agent | `MCP_SERVER_URL`, `AGENT_ENABLED_TOOLS` |
| Research graph | `RESEARCH_TIMEOUT`, `RESEARCH_RETRIEVE_TOOLS`, `RESEARCH_EXECUTE_TOOLS`, `RESEARCH_STEP_TIMEOUT_S`, `RESEARCH_MAX_STEP_TOOL_CALLS`, `RESEARCH_MAX_REPLANS` |
| SPARQL docs KB | `VECTOR_STORE_TYPE`, `DOCS_COLLECTION_NAME`, `QDRANT_HOST`, `QDRANT_PORT`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `DEFAULT_RETRIEVED_DOCS`, `SIMILARITY_THRESHOLD`, `FORCE_INDEX`, `AUTO_INIT` |
| Decision search | `SEARCH_ENDPOINT`, `EMBEDDING_API_URL`, `ELASTIC_SEARCH_K`, `ELASTIC_SEARCH_NUM_CANDIDATES`, `ELASTIC_SEARCH_MIN_SCORE`, `ELASTIC_SEARCH_KEYWORD_FIELDS`, `REQUEST_TIMEOUT`, `TITLE_FALLBACK_CHARS` |
| Discovery tools | `NOMINATIM_ENDPOINT` |
| Logging | `VERBOSE` |

All listed Agent and graph settings are passed through the typed `AgentConfig`. Streaming is not configurable: the endpoint determines whether the response is JSON or SSE.

### Endpoint configuration

`CONFIG_FILE` defaults to `/config/config.json`.

```json
{
  "endpoints": [
    {
      "endpoint_url": "https://example.org/sparql",
      "examples_file": "data/queries/shacl/examples/local_sparql_examples.ttl",
      "shapes_folder": "data/queries/shacl/shapes"
    }
  ]
}
```

If the file is missing or contains no usable endpoints, the service falls back to `MU_SPARQL_ENDPOINT` with the local VoID, examples, and SHACL paths.

## Running the Service

The supported build target is the `Dockerfile`, which extends `semtech/mu-python-template:feature-fastapi`.

```bash
docker build -t decide-research-backend .
docker run --rm -p 80:80 --env-file .env \
  -v "$(pwd)/config.json:/config/config.json:ro" \
  decide-research-backend
```

The backend also requires the services selected by its configuration:

- an LLM provider;
- a SPARQL endpoint;
- a SPARQL documentation store (in-memory or Qdrant-backed);
- mu-search/Elasticsearch and an embedding endpoint for `search_expressions`.

There is currently no canonical Compose manifest tracked in Git. Multiple ignored local manifests exist and disagree about services and tool selection; consolidating them is part of the cleanup plan.

## V3 Roadmap

V3 keeps the full research graph for structured, multi-step questions and adds two lightweight paths:

```text
POST /agent/ask --------+-> classify -> rag -------> JSON answer
                        |             -> follow-up -> JSON answer
                        |             -> research --> retrieve -> plan -> execute -> validate
                        |
POST /agent/ask/stream -+-> same modes and graph, emitted as SSE events
```

Planned endpoint contract:

| Endpoint | Scope | Transport |
|---|---|---|
| `POST /agent/ask` | Canonical `rag`, `research`, and `routed` modes | JSON only |
| `POST /agent/ask/stream` | Canonical `rag`, `research`, and `routed` modes | SSE only |
| `POST /agent/research` | Backward-compatible research mode | JSON only |
| `POST /agent/research/stream` | Backward-compatible research mode | SSE only |

The research endpoints can be renamed or removed later through a separate deprecation change; Phase 2 keeps them available.

Planned work:

1. Add `/agent/ask` and `/agent/ask/stream` with strict JSON/SSE separation.
2. Add direct RAG synthesis over `search_expressions` and a classifier for `rag`, `research`, and `followup`.
3. Add conversation-only follow-up handling.
4. Add a shared scratchpad and URI capture between research steps.
5. Add `candidate_uris` filtering to expression search.
6. Update prompts so SPARQL remains first choice for structured facts and expression search becomes a scoped content layer or explicit fuzzy fallback.

These capabilities are not yet available in the current API. See the [V3 plan](docs/research-mode-v3-plan.md), [Phase 2 implementation spec](docs/phase2-implementation-spec.md), and [cleanup audit](docs/research-mode-v3-cleanup-audit.md).

## Validation and Test State

Install the retained Streamlit application and test tooling separately from the production runtime:

```bash
pip install -r requirements-dev.txt
```

Run static and automated validation with:

```bash
python -m compileall -q src tests web.py
python -m ruff check src tests web.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The tracked suite covers JSON/SSE transport separation, graph state and routing, plan parsing, internal tool allowlisting, retained Nominatim/DuckDuckGo registration, dependency separation, and hybrid search payload validation.
