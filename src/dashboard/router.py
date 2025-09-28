from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.dashboard.dependencies import get_dashboard_service
from src.dashboard.service import DashboardService
from src.recommendations.service import RecommendationService

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/user/{user_id}", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    user_id: str,
    dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        data = await dashboard_service.get_user_dashboard_data(user_id)
        return templates.TemplateResponse(
            "user_dashboard.html", {"request": request, "data": data}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
