from fastapi import FastAPI
from contextlib import asynccontextmanager
from .db import init_db
from app.routers import ingestion

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts up
    init_db()
    yield
    # This would run when the server shuts down

app = FastAPI(title="Palm Mind Backend Assessment", version="1.0.0", lifespan=lifespan)

# Include our new router
app.include_router(ingestion.router, tags=["Ingestion"])

@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "Palm Mind backend is running!"}
