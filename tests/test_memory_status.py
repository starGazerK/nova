from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeStatusStore:
    def __init__(self, root: Path):
        self.workspace = root
        self.memory_dir = root / ".nova" / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.memory_dir.mkdir(parents=True, exist_ok=True)


class FakeStatusBackend:
    def __init__(self, *, backend: str = "mem0", healthy: bool = True, pending: int = 0):
        from nova.agent.memory_backend import MemoryBackendStatus

        self._status = MemoryBackendStatus(
            enabled=True,
            backend=backend,
            healthy=healthy,
            pending_writes=pending,
            last_error=None if healthy else "qdrant unavailable",
            last_write_at=1_780_000_000.0,
        )

    def status(self):
        return self._status


class MemoryStatusTests(unittest.TestCase):
    def test_operator_status_collects_snapshot_and_marker(self) -> None:
        from nova.agent.memory_status import get_memory_operator_status
        from nova.agent.memory_telemetry import MemoryTelemetrySnapshot

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeStatusStore(Path(tmp))
            store.memory_file.write_text("# Memory Snapshot\n", encoding="utf-8")
            marker = store.memory_dir / ".mem0_migration"
            marker.write_text("{}\n", encoding="utf-8")
            now = time.time()
            self.assertGreaterEqual(store.memory_file.stat().st_mtime, now - 5)

            with patch(
                "nova.agent.memory_status.MEMORY_TELEMETRY.snapshot",
                return_value=MemoryTelemetrySnapshot(
                    last_write_at=1_790_000_000.0,
                    last_write_scope="session",
                    last_write_mode="manual-sync",
                    last_snapshot_refresh_at=1_790_000_100.0,
                    last_snapshot_refresh_ok=True,
                    last_delete_at=1_790_000_200.0,
                    last_delete_scope="session",
                    last_delete_memory_id="mem-1",
                ),
            ):
                status = get_memory_operator_status(
                    store=store,
                    backend=FakeStatusBackend(pending=2),  # type: ignore[arg-type]
                )

            self.assertEqual(status.backend, "mem0")
            self.assertTrue(status.healthy)
            self.assertEqual(status.pending_writes, 2)
            self.assertTrue(status.migration_completed)
            self.assertIsNotNone(status.snapshot_timestamp)
            self.assertEqual(status.last_write_scope, "session")
            self.assertEqual(status.last_write_mode, "manual-sync")
            self.assertEqual(status.last_delete_memory_id, "mem-1")

    def test_render_memory_status_includes_details_and_recent_jobs(self) -> None:
        from nova.agent.memory_backend import build_memory_scope
        from nova.agent.memory_jobs import MemoryJobManager
        from nova.agent.memory_telemetry import MemoryTelemetrySnapshot
        from nova.agent.memory_status import (
            get_memory_operator_status,
            render_memory_status,
        )

        class NoopBackend:
            def add_messages(self, messages, scope, metadata):
                from nova.agent.memory_backend import MemoryWriteResult

                return MemoryWriteResult(ok=False, ids=[], error="write failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeStatusStore(Path(tmp))
            mgr = MemoryJobManager()
            job = mgr.enqueue_add(
                backend=NoopBackend(),  # type: ignore[arg-type]
                messages=[{"role": "user", "content": "remember"}],
                scope=build_memory_scope(scope="project"),
                metadata={"source": "unit_test"},
                source_cursor_start=1,
                source_cursor_end=3,
            )
            self.assertTrue(job.event.wait(2))

            with patch("nova.agent.memory_jobs.MEMORY_JOBS", mgr):
                with patch(
                    "nova.agent.memory_status.MEMORY_TELEMETRY.snapshot",
                    return_value=MemoryTelemetrySnapshot(
                        last_write_at=1_790_000_000.0,
                        last_write_scope="project",
                        last_write_mode="async-job",
                        last_snapshot_refresh_at=1_790_000_050.0,
                        last_snapshot_refresh_ok=False,
                        last_snapshot_refresh_error="snapshot failed",
                        last_delete_at=1_790_000_090.0,
                        last_delete_scope="session",
                        last_delete_memory_id="mem-9",
                    ),
                ):
                    status = get_memory_operator_status(
                        store=store,
                        backend=FakeStatusBackend(healthy=False, pending=0),  # type: ignore[arg-type]
                    )

            rendered = render_memory_status(status, detailed=True)
            self.assertIn("backend=mem0, healthy=no", rendered)
            self.assertIn("Last write scope: project", rendered)
            self.assertIn("Last write mode: async-job", rendered)
            self.assertIn("Last error: qdrant unavailable", rendered)
            self.assertIn("Last snapshot error: snapshot failed", rendered)
            self.assertIn("Last delete memory_id: mem-9", rendered)
            self.assertIn("Recent jobs:", rendered)
            self.assertIn("cursor=1..3", rendered)
            self.assertIn("error=write failed", rendered)


if __name__ == "__main__":
    unittest.main()
