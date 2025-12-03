"""Celery application configuration for async xGoal prediction tasks."""

from __future__ import annotations

import os

from celery import Celery

# Redis broker URL (default: localhost, using db 0 for broker)
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Redis backend URL for storing results (default: localhost, using db 1 for results)
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery app
celery_app = Celery(
    "xgoal_tutor",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    result_expires=3600,  # Results expire after 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Auto-discover tasks from the tasks module
celery_app.autodiscover_tasks(["xgoal_tutor.api"])
