"""
nova/agent/memory_telemetry.py - Shared operator telemetry for memory actions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryTelemetrySnapshot:
    last_write_at: float | None = None
    last_write_scope: str | None = None
    last_write_mode: str | None = None
    last_snapshot_refresh_at: float | None = None
    last_snapshot_refresh_ok: bool | None = None
    last_snapshot_refresh_error: str | None = None
    last_delete_at: float | None = None
    last_delete_scope: str | None = None
    last_delete_memory_id: str | None = None


class MemoryTelemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MemoryTelemetrySnapshot()

    def snapshot(self) -> MemoryTelemetrySnapshot:
        with self._lock:
            return self._snapshot

    def record_write(
        self,
        *,
        scope: str | None,
        mode: str,
        when: float | None = None,
    ) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = MemoryTelemetrySnapshot(
                last_write_at=when or time.time(),
                last_write_scope=scope,
                last_write_mode=mode,
                last_snapshot_refresh_at=current.last_snapshot_refresh_at,
                last_snapshot_refresh_ok=current.last_snapshot_refresh_ok,
                last_snapshot_refresh_error=current.last_snapshot_refresh_error,
                last_delete_at=current.last_delete_at,
                last_delete_scope=current.last_delete_scope,
                last_delete_memory_id=current.last_delete_memory_id,
            )

    def record_snapshot_refresh(
        self,
        *,
        ok: bool,
        error: str | None = None,
        when: float | None = None,
    ) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = MemoryTelemetrySnapshot(
                last_write_at=current.last_write_at,
                last_write_scope=current.last_write_scope,
                last_write_mode=current.last_write_mode,
                last_snapshot_refresh_at=when or time.time(),
                last_snapshot_refresh_ok=ok,
                last_snapshot_refresh_error=None if ok else error,
                last_delete_at=current.last_delete_at,
                last_delete_scope=current.last_delete_scope,
                last_delete_memory_id=current.last_delete_memory_id,
            )

    def record_delete(
        self,
        *,
        memory_id: str,
        scope: str | None,
        when: float | None = None,
    ) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = MemoryTelemetrySnapshot(
                last_write_at=current.last_write_at,
                last_write_scope=current.last_write_scope,
                last_write_mode=current.last_write_mode,
                last_snapshot_refresh_at=current.last_snapshot_refresh_at,
                last_snapshot_refresh_ok=current.last_snapshot_refresh_ok,
                last_snapshot_refresh_error=current.last_snapshot_refresh_error,
                last_delete_at=when or time.time(),
                last_delete_scope=scope,
                last_delete_memory_id=memory_id,
            )


MEMORY_TELEMETRY = MemoryTelemetry()
