"""YAAP — Yet Another Annotation Platform. FastAPI application."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routers import annotations, datasets, images, models, projects, sam, training

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="YAAP", version="1.0.0",
              description="Yet Another Annotation Platform — local, open, ultralytics-ready.")

init_db()

for r in (projects.router, images.router, annotations.router,
          models.router, datasets.router, training.router, sam.router):
    app.include_router(r)

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
