import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import builtins
from src.job import startup_tasks
from src.api import router, lifespan, mount_mcp

from helpers import logger, query, update

# The semtech/mu-python-template injects the 'app' instance. We retrieve it here safely.
app: FastAPI = getattr(builtins, "app", FastAPI())

# Attach the lifespan handler to the existing app
app.router.lifespan_context = lifespan

# Include the router with our endpoints
app.include_router(router)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mount_mcp(app)
