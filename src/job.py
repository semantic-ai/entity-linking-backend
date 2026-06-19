import threading

from helpers import logger

from decide_ai_service_base.util import (
    TaskProcessor,
    fail_busy_and_scheduled_tasks,
    wait_for_triplestore,
    write_agent_info
)

import src.task  # noqa: F401 — registers NamedEntityLinkingTask as a Task subclass

_lock = threading.Lock()
_processor = TaskProcessor(lock=_lock)


def startup_tasks() -> None:
    """Boot sequence: wait for the triplestore, fail leftover busy tasks, drain the queue."""
    wait_for_triplestore()
    logger.info("Writing agent info...")
    write_agent_info("http://lblod.data.gift/id/components/named-entity-linking/v1.0.0")
    
    logger.info("Failing leftover busy/scheduled tasks...")
    fail_busy_and_scheduled_tasks()
    logger.info("Processing open tasks...")
    _processor()


def process_open_tasks() -> None:
    """Called from /delta background tasks; drains the open-task queue once."""
    _processor()
