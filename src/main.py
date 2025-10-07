from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.dashboard.router import router as dashboard_router
from src.database import close_mongo_connection, connect_to_mongo
from src.recommendations.router import calculate_user_pair_match
from src.recommendations.router import router as recommendations_router
from src.recommendations.schemas import UserUserMatchResponse


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
    docs_url="/api-docs",  # Move Swagger UI here for jugadding
    redoc_url="/api-redoc",
)

# Include routers
app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
app.include_router(
    recommendations_router, prefix="/recommendations", tags=["recommendations"]
)

app.add_api_route(
    "/docs/recommendations/recommendations/match/{user_a_id}/{user_b_id}",
    calculate_user_pair_match,
    methods=["GET"],
    response_model=UserUserMatchResponse,
    tags=["recommendations"],
    summary="Calculate match score between two specific users",
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
