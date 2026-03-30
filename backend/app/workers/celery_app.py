"""
Celery Application
==================
Manages the task queue for long-running, resource-intensive operations
(LLM fine-tuning) that must not run inside the FastAPI request handlers.

Configuration
-------------
Broker and result backend both point to Redis.  The connection URL is
read from settings.REDIS_URL so it can be overridden per environment.

Worker concurrency
------------------
Fine-tuning is GPU/memory bound — running more than one job at a time
typically exhausts VRAM.  The worker is configured with:

    --concurrency 1   (set in docker-compose command)

This means at most one training job runs at a time.  Additional jobs
wait in the Redis queue until the worker is free.

Task time limits
----------------
  soft_time_limit = 4 hours   → raises SoftTimeLimitExceeded so the task
                                  can clean up and mark the job FAILED
  time_limit      = 4.5 hours → hard SIGKILL fallback

Routing
-------
All tasks go to the "fine_tune" queue, which the worker consumes
exclusively.  Future task types (e.g. OCR, embedding) can be added
to separate queues without changing the fine-tune worker.
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from app.config import settings

celery_app = Celery(
    "ehr_ai_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.fine_tune_task"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Reliability
    task_acks_late=True,           # ack only after successful execution
    worker_prefetch_multiplier=1,  # one task at a time per worker process

    # Time limits (seconds) — prevents GPU jobs running indefinitely
    task_soft_time_limit=4 * 3600,    # 4 hours  → raises SoftTimeLimitExceeded
    task_time_limit=int(4.5 * 3600),  # 4.5 hrs  → hard kill

    # Result expiry — keep task results for 48 hours for status polling
    result_expires=48 * 3600,

    # Routing
    task_default_queue="fine_tune",
    task_queues=[
        Queue("fine_tune"),
    ],
)
