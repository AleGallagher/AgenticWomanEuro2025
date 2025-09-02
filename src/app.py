import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.errors.handlers import register_exception_handlers
from config.logging_config import setup_logging
from routers import agent, feed_back

load_dotenv()

logger = logging.getLogger(__name__)
app = FastAPI(
    title="UEFA Women's EURO 2025 Assistant",
    description="An AI assistant for UEFA Women's EURO 2025",
    version="1.0.0",
    openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_logging()
register_exception_handlers(app)
app.include_router(agent.router)
app.include_router(feed_back.router)

@app.get("/")
async def root():
    return {"greeting": "Hello UEFA Women's EURO 2025"}
