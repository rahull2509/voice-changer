"""FastAPI application entry point for VaniFlow Voice Changer."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router, set_pipeline
from app.audio.pipeline import AudioPipeline
from app.config import AppConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vaniflow")

# Global pipeline instance
config = AppConfig.load("config.json")
pipeline = AudioPipeline(config)
set_pipeline(pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager ensuring clean startup and audio shutdown."""
    logger.info("VaniFlow Voice Changer starting up...")
    logger.info("Host: %s:%d, Default Mode: %s", config.host, config.port, config.mode)
    yield
    logger.info("Shutting down VaniFlow Voice Changer...")
    if pipeline.status.is_running:
        pipeline.stop()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="VaniFlow Voice Changer",
    description="Local Real-Time Voice Conversion for Windows (CPU)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount UI static files
ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


@app.get("/")
def serve_index():
    """Serve the web control panel UI."""
    index_file = ui_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "VaniFlow backend running. UI not found at app/ui/index.html"}


def main():
    """Launch backend server."""
    port = int(os.getenv("PORT", config.port))
    host = os.getenv("HOST", config.host)
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
