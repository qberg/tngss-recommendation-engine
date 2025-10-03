from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.dashboard.router import router as dashboard_router
from src.database import close_mongo_connection, connect_to_mongo
from src.recommendations.router import router as recommendations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


# Create FastAPI app
app = FastAPI(
    title="Recommendation Service API",
    description="Multi-vector recommendation system for TNGSS",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
app.include_router(
    recommendations_router, prefix="/recommendations", tags=["recommendations"]
)


@app.get("/")
async def root():
    return {
        "message": "Recommendation Service API",
        "dashboard": "/dashboard",
        "recommendations": "/recommendations",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
