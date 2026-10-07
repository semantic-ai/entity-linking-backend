import os

# Integration tests need the service image and Virtuoso. Run them through docker-compose.test.yml.
collect_ignore = [] if os.getenv("INTEGRATION_TESTS") else ["integration"]
