import os
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

NEL_TASK_OPERATION = "http://lblod.data.gift/id/jobs/concept/TaskOperation/named-entity-linking"

CONFIG_FILE = Path(os.getenv("CONFIG_FILE", "/config/config.json"))

_file_config: dict = {}
if CONFIG_FILE.exists():
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as _f:
            _file_config = json.load(_f)
    except Exception as e:
        print(f"Error loading config from {CONFIG_FILE}: {e}")


class Settings(BaseModel):
    """Service configuration."""

    # Search
    search_endpoint: str = os.getenv("SEARCH_ENDPOINT", "http://search")

    nominatim_endpoint: str = os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org/")

    # Stack
    mu_sparql_endpoint: str = os.getenv("MU_SPARQL_ENDPOINT", "http://virtuoso:8890/sparql")

    linking_job_type: str = os.getenv(
        "LINKING_JOB_TYPE",
        "http://lblod.data.gift/id/jobs/concept/JobType/entity-linking",
    )
    resource_base: str = os.getenv("RESOURCE_BASE", "http://data.lblod.info/id/")

    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))


settings = Settings()


def _normalize_overrides(raw) -> dict:
    """Casefold + strip keys of a flat {entity_label: override} mapping; expand
    each entry's `aliases` list as extra keys pointing at the same override."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        aliases = v.get("aliases", []) or []
        clean_v = {kk: vv for kk, vv in v.items() if kk != "aliases"}
        out[k.strip().casefold()] = clean_v
        for alias in aliases:
            if isinstance(alias, str):
                out[alias.strip().casefold()] = clean_v
    return out


location_overrides: dict = _normalize_overrides(_file_config.get("location_overrides"))
