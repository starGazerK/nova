"""
nova/agent/memory_jobs.py - In-process memory write jobs.

This queue is for Python memory operations, not shell commands. It lets Dream
schedule Mem0 writes without blocking the user-facing turn.
"""

from __future__ import annotations

import threading
import time
import uuid
from queue import Queue
from dataclasses import dataclass, field
from typing import Any, Callable

from nova.agent.memory_backend import MemoryBackend, MemoryScope, MemoryWriteResult


@dataclass
class MemoryWriteJob:
    job_id: str
    status: str
    scope: MemoryScope
    messages: list[dict]
    metadata: dict[str, Any]
    source_cursor_start: int | None
    source_cursor_end: int | None
    started_at: float
    finished_at: float | None = None
    ids: list[str] = field(default_factory=list)
    error: str | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)


class MemoryJobManager:
    """Threaded queue for memory writes."""

    def __init__(self):
        self._jobs: dict[str, MemoryWriteJob] = {}
        self._lock = threading.Lock()
        self._queue: Queue[tuple[str, MemoryBackend, Callable[[MemoryWriteResult], None] | None]] = Queue()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="nova-memory-writer",
            daemon=True,
        )
        self._worker.start()

    def enqueue_add(
        self,
        *,
        backend: MemoryBackend,
        messages: list[dict],
        scope: MemoryScope,
        metadata: dict[str, Any],
        source_cursor_start: int | None = None,
        source_cursor_end: int | None = None,
        on_success: Callable[[MemoryWriteResult], None] | None = None,
    ) -> MemoryWriteJob:
        job = MemoryWriteJob(
            job_id=str(uuid.uuid4())[:8],
            status="queued",
            scope=scope,
            messages=[dict(message) for message in messages],
            metadata=dict(metadata),
            source_cursor_start=source_cursor_start,
            source_cursor_end=source_cursor_end,
            started_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put((job.job_id, backend, on_success))
        return job

    def _worker_loop(self) -> None:
        while True:
            job_id, backend, on_success = self._queue.get()
            try:
                self._run_add(job_id, backend, on_success)
            finally:
                self._queue.task_done()

    def _run_add(
        self,
        job_id: str,
        backend: MemoryBackend,
        on_success: Callable[[MemoryWriteResult], None] | None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
        try:
            result = backend.add_messages(job.messages, job.scope, job.metadata)
            if result.ok:
                if on_success:
                    on_success(result)
                with self._lock:
                    job.status = "completed"
                    job.ids = list(result.ids)
                    job.error = None
                    job.finished_at = time.time()
                    job.event.set()
            else:
                with self._lock:
                    job.status = "error"
                    job.error = result.error or "memory write failed"
                    job.finished_at = time.time()
                    job.event.set()
        except Exception as exc:
            with self._lock:
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()
                job.event.set()

    def get(self, job_id: str) -> MemoryWriteJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[MemoryWriteJob]:
        with self._lock:
            return sorted(
                list(self._jobs.values()),
                key=lambda job: job.started_at,
                reverse=True,
            )

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status in {"queued", "running"})


MEMORY_JOBS = MemoryJobManager()
