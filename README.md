# Entity Linking Backend MCP Server

This repository contains the backend service for the Entity Linking Service. It exposes a Model Context Protocol (MCP) server that provides tools for querying SPARQL endpoints, searching locations, performing web searches, and utilizing a vector-based knowledge base (for example sparql queries).

## Project Structure

```
entity-linking-backend/
├── config/             # Configuration files
├── data/               # Data files (metadata, examples, shapes)
├── src/                # Source code
│   ├── agent.py        # Agent implementation
│   ├── api.py          # FastAPI entry point
│   ├── mcp_server.py   # MCP Server definition
│   ├── knowledge_base.py # Qdrant knowledge base integration
│   ├── tools/          # Tool implementations (SPARQL, Nominatim, Web)
│   └── utils/          # Utility functions
└── requirements.txt    # Python dependencies
```

## Features

- **SPARQL Integration**: Tools to generate and execute SPARQL queries against configured endpoints.
- **Knowledge Base**: Uses Qdrant and FastEmbed/Ollama for semantic search over documentation and examples. (can be used without Qdrant in memory)
- **Location Search**: Integration with Nominatim for geocoding.
- **Web Search**: DuckDuckGo search integration for web searches without API keys.
- **Multiple LLM Support**: Configurable to use OpenAI, Mistral, or Ollama.

## Available Tools

The following tools are available via the MCP server:

- **search_location**: Search for a location (entity linking) based on a query, city, and country. Returns the nominatim reponse.
- **search_web**: Search the web for additional information. Useful for general knowledge questions about persons, places, events, etc.
- **search_sparql_docs**: Assist users in writing SPARQL queries to access resources by retrieving relevant examples and classes schema.
- **execute_sparql_query**: Execute a SPARQL query against a SPARQL endpoint.

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

The application is configured via environment variables:

```env
# LLM Provider (openai, mistral, ollama)
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4

# Mistral Configuration
MISTRAL_API_KEY=your_key_here
MISTRAL_MODEL=mistral-medium

# Services
QDRANT_HOST=localhost
QDRANT_PORT=6333
NOMINATIM_ENDPOINT=http://localhost:8080/
OLLAMA_HOST=http://localhost:11434

# MCP Configuration
MCP_SERVER_URL=http://localhost:80/mcp/sse
ENABLED_TOOLS=search_sparql_docs,execute_sparql_query,search_web,search_location
```

## Switching Providers & Tool Selection

The application supports multiple LLM providers including OpenAI, Mistral, and Ollama (local).

### Local Execution with Ollama

You can run the agent locally using Ollama. This is useful for privacy or cost reasons.
To use Ollama, set `LLM_PROVIDER=ollama` and configure the endpoints and models in the env.

Testing using following local models:
- **Mistral Nemo**: Decent performance, functional tool-calling.
- **Ministral-3:14b (instruct)**:  Issue with tool-calling via ollama, running via Minstral API achieves best results.

### Tool Selection for Small Models

When using smaller local models (like 7B or 12B parameter models), it is highly recommended to **limit the number of enabled tools**. Smaller models can struggle with reasoning when presented with too many tools or irrelevant context.

- **For Location Queries**: If you are only interested in geocoding or finding places (e.g., using Nominatim), only enable the `search_location` tool.
- **For Mandataries/Administrative Bodies**: If you are querying SPARQL endpoints for government officials or bodies, enable only the SPARQL tools (`search_sparql_docs`, `execute_sparql_query`).

You can control this via the `ENABLED_TOOLS` environment variable:

```env
# Only for specialized location tasks
ENABLED_TOOLS=search_location

# Only for SPARQL/Knowledge Base tasks
ENABLED_TOOLS=search_sparql_docs,execute_sparql_query
```

## Usage

### Running Locally (HTTP API)

To run the HTTP server which exposes the MCP SSE endpoint:

```bash
python -m src.api
```

The server will start on `http://0.0.0.0:80`. The MCP SSE endpoint is available at `/mcp/sse`.
 
## Docker Compose Example

A minimal `docker-compose.yml` for running the service alongside Qdrant, Nominatim and Ollama:

```yaml
version: '3.8'

services:
    decide-mcp:
        build: .
        volumes:
            - ./:/app
        ports:
            - "80:80"
        env_file:
            - .env
        environment:
            - QDRANT_HOST=qdrant
            - QDRANT_PORT=6333
            - OLLAMA_HOST=http://ollama:11434
            - NOMINATIM_ENDPOINT=http://nominatim:8080/
            - MCP_SERVER_URL=http://localhost:80/mcp/sse
            - ENABLED_TOOLS=search_sparql_docs,execute_sparql_query
            - ...
        depends_on:
            - qdrant
            - nominatim
            - ollama

    qdrant:
        image: qdrant/qdrant
        ports:
            - "6333:6333"

    nominatim:
        image: mediagis/nominatim:4.2
        ports:
            - "8080:8080"

    ollama:
        image: ollama/ollama:latest
        ports:
            - "11434:11434"
```

Start the stack with:

```bash
docker compose up 
```

## API Endpoints

This service exposes a small HTTP API (FastAPI). Two commonly used endpoints are shown below.

- **Health check — GET /**

    Request:

    ```bash
    curl -s http://localhost/ | jq
    ```

    Example response:

    ```json
    {
        "status": "running",
        "endpoints": ["/mcp"]
    }
    ```

- **Agent Endpoints**

    For quick testing you can also call the agent HTTP endpoints directly.

    **Free-form Query — `POST /agent/query`**

    ```bash
    curl -X POST http://localhost/agent/query \
        -H "Content-Type: application/json" \
        -d '{"query": "Return the openstreetmaps URI of location 'Station Gent-Sint-Pieters'. Keep searching untill you find closest match."}'
    ```

    **Structured Query — `POST /agent/query_structured`**

    Target specific entity classes. Currently supported classes include **Mandatary** and **Administrative Body**.

    ```bash
    curl -X POST http://localhost/agent/query_structured \
        -H "Content-Type: application/json" \
        -d '{"entity_class": "Administrative Body", "entity_label": "Vast Bureau", "location": "Gent"}'
    ```

    - **MCP SSE endpoint — `/mcp/sse`**

    The MCP server is mounted under `/mcp`. To open a Server-Sent Events (SSE) stream use:

    ```bash
    curl -N -H "Accept: text/event-stream" http://localhost/mcp/sse
    ```

    The exact event format depends on the MCP client/server interaction. For interactive usage, connect an MCP-capable client (or use the `fastmcp` client) and exchange the MCP messages over the SSE transport.

