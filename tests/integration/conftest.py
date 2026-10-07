import http.server
import os
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from .tasks import FIXTURES

ENDPOINT = os.environ["MU_SPARQL_ENDPOINT"]

SERVICES_PORT = 8001

# mu-search and Nominatim answers, by request path.
RESPONSES = {
    "/organizations/search/": "organizations.json",
    "/search": "nominatim_search.json",
    "/lookup": "nominatim_lookup.json",
}


@pytest.fixture(scope="session", autouse=True)
def virtuoso():
    deadline = time.monotonic() + 180
    while True:
        try:
            requests.post(
                ENDPOINT, data={"query": "ASK { ?s ?p ?o }"}, timeout=10
            ).raise_for_status()
            return
        except requests.RequestException:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


@pytest.fixture(scope="session")
def search_services():
    """Answer mu-search and Nominatim on 127.0.0.1:8001; yields the (path, params) received."""
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            received.append((url.path, parse_qs(url.query)))
            body = (FIXTURES / RESPONSES[url.path]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", SERVICES_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield received
    server.shutdown()
