from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.dashboard.router import router as dashboard_router
from src.database import connect_to_mongo

# Add other routers as you create them
# from src.recommendations.router import router as recommendations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield


# Create FastAPI app
app = FastAPI(
    title="Recommendation Service API",
    description="Multi-vector recommendation system for TNGSS",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])

# Add other routers here as needed
# app.include_router(recommendations_router, prefix="/api/recommendations", tags=["recommendations"])

# Mount static files if you need them later
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Recommendation Service API", "dashboard": "/dashboard"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
