from src.dashboard.service import DashboardService
from src.database import get_database
from src.recommendations.service import RecommendationService


async def get_dashboard_service() -> DashboardService:
    """Dependency to get dashboard service instance."""
    db = get_database()
    rec_service = RecommendationService(db)
    return DashboardService(rec_service)
