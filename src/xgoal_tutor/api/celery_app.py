"""Celery application used by the asynchronous summary endpoints."""

from __future__ import annotations

import os

from celery import Celery

_CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost//")


def create_celery_app() -> Celery:
    """Instantiate and configure the Celery application."""

    app = Celery("xgoal", broker=_CELERY_BROKER_URL)
    app.conf.task_default_queue = "xgoal"
    app.conf.task_routes = {
        "xgoal.match_summary": {"queue": "xgoal"},
        "xgoal.player_summary": {"queue": "xgoal"},
    }
    return app


app = create_celery_app()
