import asyncio
import concurrent.futures
import contextlib
from typing import AsyncIterator

from fastapi import FastAPI, APIRouter, BackgroundTasks

from src.job import process_open_tasks, startup_tasks
from decide_ai_service_base.schema import NotificationResponse

from helpers import logger


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that runs startup tasks."""
    logger.info("Running startup tasks...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(startup_tasks)
        await asyncio.wrap_future(future)

    yield


router = APIRouter()


@router.get("/")
async def health():
    return {"status": "running"}


@router.post("/delta", status_code=202)
def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    logger.info("Received delta notification")
    background_tasks.add_task(process_open_tasks)
    return NotificationResponse(
        status="accepted",
        message="Processing started",
    )