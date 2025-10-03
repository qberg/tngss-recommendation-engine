"""Celery application configuration."""

from celery import Celery

from src.config import settings

celery_app = Celery(
    "recommendations",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["src.recommendations.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
)

if __name__ == "__main__":
    print("[INFO] Testing Celery configuration...")
    print(f"Broker: {celery_app.conf.broker_url}")
    print(f"Backend: {celery_app.conf.result_backend}")

    try:
        celery_app.connection().connect()
        print("[SUCCESS] Successfully connected to Redis broker")
    except Exception as e:
        print(f"[Failed] to connect: {e}")
