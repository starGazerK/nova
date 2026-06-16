"""
nova/agent/memory_status.py - Operator visibility for memory backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nova.agent.memory_backend import (
    MemoryBackend,
    MemoryBackendStatus,
    create_memory_backend,
)
from nova.agent.memory_telemetry import MEMORY_TELEMETRY


@dataclass(frozen=True)
class MemoryJobSnapshot:
    job_id: str
    status: str
    scope: str
    source_cursor_start: int | None
    source_cursor_end: int | None
    started_at: float
    finished_at: float | None
    error: str | None


@dataclass(frozen=True)
class MemoryOperatorStatus:
    backend: str
    healthy: bool
    pending_writes: int
    last_error: str | None
    last_write_at: float | None
    last_write_scope: str | None
    last_write_mode: str | None
    snapshot_path: str
    snapshot_timestamp: float | None
    last_snapshot_refresh_at: float | None
    last_snapshot_refresh_ok: bool | None
    last_snapshot_refresh_error: str | None
    last_delete_at: float | None
    last_delete_scope: str | None
    last_delete_memory_id: str | None
    migration_marker_path: str
    migration_completed: bool
    recent_jobs: list[MemoryJobSnapshot] = field(default_factory=list)


def _snapshot_timestamp(path: Path) -> float | None:
    try:
        if path.exists():
            return path.stat().st_mtime
    except OSError:
        return None
    return None


def _format_time(value: float | None) -> str:
    if value is None:
        return "none"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))
    except (OSError, OverflowError, ValueError):
        return "invalid"


def _job_snapshots(limit: int = 5) -> list[MemoryJobSnapshot]:
    try:
        from nova.agent.memory_jobs import MEMORY_JOBS

        jobs = MEMORY_JOBS.list()[:limit]
    except Exception:
        return []
    return [
        MemoryJobSnapshot(
            job_id=job.job_id,
            status=job.status,
            scope=job.scope.scope,
            source_cursor_start=job.source_cursor_start,
            source_cursor_end=job.source_cursor_end,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
        )
        for job in jobs
    ]


def get_memory_operator_status(
    *,
    store=None,
    backend: MemoryBackend | None = None,
) -> MemoryOperatorStatus:
    """Collect memory state for `/status` and `/memory status`."""
    from nova.agent.memory import _STORE

    target_store = store or _STORE
    target_backend = backend or create_memory_backend(target_store)
    backend_status: MemoryBackendStatus = target_backend.status()
    telemetry = MEMORY_TELEMETRY.snapshot()
    snapshot_path = target_store.memory_file
    migration_marker = target_store.memory_dir / ".mem0_migration"
    return MemoryOperatorStatus(
        backend=backend_status.backend,
        healthy=backend_status.healthy,
        pending_writes=backend_status.pending_writes,
        last_error=backend_status.last_error,
        last_write_at=telemetry.last_write_at or backend_status.last_write_at,
        last_write_scope=telemetry.last_write_scope,
        last_write_mode=telemetry.last_write_mode,
        snapshot_path=str(snapshot_path),
        snapshot_timestamp=_snapshot_timestamp(snapshot_path),
        last_snapshot_refresh_at=telemetry.last_snapshot_refresh_at,
        last_snapshot_refresh_ok=telemetry.last_snapshot_refresh_ok,
        last_snapshot_refresh_error=telemetry.last_snapshot_refresh_error,
        last_delete_at=telemetry.last_delete_at,
        last_delete_scope=telemetry.last_delete_scope,
        last_delete_memory_id=telemetry.last_delete_memory_id,
        migration_marker_path=str(migration_marker),
        migration_completed=migration_marker.exists(),
        recent_jobs=_job_snapshots(),
    )


def render_memory_status_lines(status: MemoryOperatorStatus, *, detailed: bool = False) -> list[str]:
    """Render status lines without Rich markup so callers can style them."""
    health = "yes" if status.healthy else "no"
    lines = [
        f"Memory : backend={status.backend}, healthy={health}, pending={status.pending_writes}",
    ]
    if detailed:
        lines.extend([
            f"Last write: {_format_time(status.last_write_at)}",
            f"Last write scope: {status.last_write_scope or 'none'}",
            f"Last write mode: {status.last_write_mode or 'none'}",
            f"Last error: {status.last_error or 'none'}",
            f"Snapshot : {_format_time(status.snapshot_timestamp)} ({status.snapshot_path})",
            (
                f"Last snapshot refresh: {_format_time(status.last_snapshot_refresh_at)} "
                f"(ok={'yes' if status.last_snapshot_refresh_ok else 'no' if status.last_snapshot_refresh_ok is False else 'unknown'})"
            ),
            f"Last snapshot error: {status.last_snapshot_refresh_error or 'none'}",
            f"Last delete: {_format_time(status.last_delete_at)}",
            f"Last delete scope: {status.last_delete_scope or 'none'}",
            f"Last delete memory_id: {status.last_delete_memory_id or 'none'}",
            (
                "Migration: completed"
                if status.migration_completed
                else f"Migration: not completed ({status.migration_marker_path})"
            ),
        ])
        if status.recent_jobs:
            lines.append("Recent jobs:")
            for job in status.recent_jobs:
                cursor = "-"
                if job.source_cursor_start is not None or job.source_cursor_end is not None:
                    cursor = f"{job.source_cursor_start or '-'}..{job.source_cursor_end or '-'}"
                suffix = f", error={job.error}" if job.error else ""
                lines.append(
                    f"- {job.job_id} {job.status} scope={job.scope} cursor={cursor}{suffix}"
                )
        else:
            lines.append("Recent jobs: none")
    return lines


def render_memory_status(status: MemoryOperatorStatus, *, detailed: bool = False) -> str:
    return "\n".join(render_memory_status_lines(status, detailed=detailed))
