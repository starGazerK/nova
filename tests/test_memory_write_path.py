from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeWriteBackend:
    def __init__(self, *, ok: bool = True):
        from nova.agent.memory_backend import MemoryWriteResult

        self.ok = ok
        self.calls: list[tuple[list[dict], object, dict]] = []
        self._result_cls = MemoryWriteResult

    def add_messages(self, messages, scope, metadata):
        self.calls.append((messages, scope, metadata))
        if self.ok:
            return self._result_cls(ok=True, ids=["mem-1"])
        return self._result_cls(ok=False, ids=[], error="write failed")


class FakeGit:
    def __init__(self):
        self.commits: list[str] = []

    def auto_commit(self, message: str) -> bool:
        self.commits.append(message)
        return True


class FakeStore:
    def __init__(self, archived_entries: list[dict] | None = None):
        self.workspace = Path(tempfile.gettempdir()) / "nova-memory-test"
        self.archived_entries = archived_entries or []
        self.cursor: int = 0
        self.compacted = False
        self.git_initialized = False
        self.git = FakeGit()

    def read_unprocessed_history(self, since_cursor: int) -> list[dict]:
        return [entry for entry in self.archived_entries if entry["cursor"] > since_cursor]

    def get_last_dream_cursor(self) -> int:
        return self.cursor

    def set_last_dream_cursor(self, cursor: int) -> None:
        self.cursor = cursor

    def compact_history(self) -> None:
        self.compacted = True

    def ensure_git_initialized(self) -> bool:
        self.git_initialized = True
        return True

    def read_user(self) -> str:
        return "(empty)"

    def read_soul(self) -> str:
        return "(empty)"

    def read_memory(self) -> str:
        return "(empty)"


class TrackingDreamProcessor:
    pass


class MemoryWritePathTests(unittest.TestCase):
    def test_memory_job_success_and_failure_cursor_behavior(self) -> None:
        from nova.agent.memory_backend import build_memory_scope
        from nova.agent.memory_jobs import MemoryJobManager

        scope = build_memory_scope(scope="project")
        success_backend = FakeWriteBackend(ok=True)
        success_called: list[bool] = []
        mgr = MemoryJobManager()

        success_job = mgr.enqueue_add(
            backend=success_backend,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "remember success"}],
            scope=scope,
            metadata={"source": "unit_test"},
            on_success=lambda _result: success_called.append(True),
        )
        self.assertTrue(success_job.event.wait(2))
        self.assertEqual(success_job.status, "completed")
        self.assertEqual(success_job.ids, ["mem-1"])
        self.assertEqual(success_called, [True])

        failure_backend = FakeWriteBackend(ok=False)
        failure_called: list[bool] = []
        failure_job = mgr.enqueue_add(
            backend=failure_backend,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "remember failure"}],
            scope=scope,
            metadata={"source": "unit_test"},
            on_success=lambda _result: failure_called.append(True),
        )
        self.assertTrue(failure_job.event.wait(2))
        self.assertEqual(failure_job.status, "error")
        self.assertEqual(failure_job.error, "write failed")
        self.assertEqual(failure_called, [])

    def test_mem0_mode_queues_write_and_skips_markdown_phases(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            phase1_called = False
            phase2_called = False

            async def _phase1_analyze(self, *args, **kwargs):
                self.phase1_called = True
                return "[MEMORY] stable project fact"

            async def _phase2_execute(self, *args, **kwargs):
                self.phase2_called = True
                return [{"name": "edit_file"}]

        store = FakeStore([
            {"cursor": 1, "timestamp": "2026-06-08 10:00", "content": "User: remember alpha"}
        ])
        backend = FakeWriteBackend(ok=True)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
        ):
            result = asyncio.run(processor.run([], session_key="session_1"))

        self.assertTrue(result)
        jobs = mgr.list()
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertTrue(job.event.wait(2))
            self.assertEqual(job.status, "completed")
        self.assertEqual(store.cursor, 1)
        self.assertTrue(processor.phase1_called)
        self.assertFalse(processor.phase2_called)
        self.assertFalse(store.git_initialized)
        self.assertEqual([call[1].scope for call in backend.calls], ["session", "project"])
        self.assertEqual(backend.calls[0][2]["memory_kind"], "session_conversation")
        self.assertEqual(backend.calls[1][2]["memory_kind"], "project_fact")

    def test_mem0_write_failure_does_not_advance_cursor(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

        store = FakeStore([
            {"cursor": 2, "timestamp": "2026-06-08 10:00", "content": "User: remember beta"}
        ])
        backend = FakeWriteBackend(ok=False)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
        ):
            result = asyncio.run(processor.run([], session_key="session_1"))

        self.assertTrue(result)
        jobs = mgr.list()
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertTrue(job.event.wait(2))
            self.assertEqual(job.status, "error")
        self.assertEqual(store.cursor, 0)
        self.assertFalse(store.compacted)

    def test_hybrid_mode_queues_mem0_and_keeps_markdown_phase(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

            async def _phase2_execute(self, *args, **kwargs):
                return [{"name": "edit_file", "status": "ok", "detail": "Successfully edited"}]

        store = FakeStore([
            {"cursor": 3, "timestamp": "2026-06-08 10:00", "content": "User: remember gamma"}
        ])
        backend = FakeWriteBackend(ok=True)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "hybrid"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
        ):
            result = asyncio.run(processor.run([], session_key="session_1"))

        self.assertTrue(result)
        jobs = mgr.list()
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertTrue(job.event.wait(2))
            self.assertEqual(job.status, "completed")
        self.assertTrue(store.git_initialized)
        self.assertEqual(store.cursor, 3)
        self.assertTrue(store.compacted)
        self.assertEqual([call[1].scope for call in backend.calls], ["session", "project"])

    def test_force_run_bypasses_signal_threshold_and_waits_for_mem0(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

        store = FakeStore()
        backend = FakeWriteBackend(ok=True)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]
        messages = [
            {"role": "user", "content": "remember this"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
        ):
            skipped = asyncio.run(processor.run(messages, session_key="session_1"))
            forced = asyncio.run(processor.run(messages, session_key="session_1", force=True, wait_for_mem0=True))

        self.assertFalse(skipped)
        self.assertTrue(forced)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(store.cursor, 0)

    def test_wait_for_mem0_returns_false_when_write_fails(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

        store = FakeStore()
        backend = FakeWriteBackend(ok=False)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]
        messages = [
            {"role": "user", "content": "remember this"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
        ):
            result = asyncio.run(processor.run(messages, session_key="session_1", force=True, wait_for_mem0=True))

        self.assertFalse(result)

    def test_sync_mem0_write_bypasses_job_queue_and_refreshes_snapshot(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

        store = FakeStore([
            {"cursor": 4, "timestamp": "2026-06-08 10:00", "content": "User: remember delta"}
        ])
        backend = FakeWriteBackend(ok=True)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]
        messages = [
            {"role": "user", "content": "remember this now"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
            patch("nova.agent.memory_snapshot.refresh_memory_snapshot_silently") as refresh,
        ):
            result = asyncio.run(processor.run(
                messages,
                session_key="session_1",
                force=True,
                sync_mem0=True,
            ))

        self.assertTrue(result)
        self.assertEqual(mgr.list(), [])
        self.assertEqual(store.cursor, 4)
        self.assertEqual([call[1].scope for call in backend.calls], ["session", "project"])
        self.assertEqual(backend.calls[0][0], [{"role": "user", "content": "remember this now"}])
        self.assertFalse(backend.calls[0][2]["mem0_infer"])
        self.assertFalse(backend.calls[1][2]["mem0_infer"])
        refresh.assert_called_once()

        from nova.agent.memory_telemetry import MEMORY_TELEMETRY

        telemetry = MEMORY_TELEMETRY.snapshot()
        self.assertEqual(telemetry.last_write_scope, "project")
        self.assertEqual(telemetry.last_write_mode, "manual-sync")
        self.assertTrue(telemetry.last_snapshot_refresh_ok is None or telemetry.last_snapshot_refresh_ok)

    def test_sync_mem0_write_failure_does_not_advance_cursor(self) -> None:
        from nova.agent import memory, memory_jobs

        class Processor(memory.DreamProcessor):
            async def _phase1_analyze(self, *args, **kwargs):
                return "[MEMORY] stable project fact"

        store = FakeStore([
            {"cursor": 5, "timestamp": "2026-06-08 10:00", "content": "User: remember epsilon"}
        ])
        backend = FakeWriteBackend(ok=False)
        mgr = memory_jobs.MemoryJobManager()
        processor = Processor(store, provider=object(), emit_output=False)  # type: ignore[arg-type]
        messages = [
            {"role": "user", "content": "remember this now"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(memory, "NOVA_MEMORY_BACKEND", "mem0"),
            patch("nova.agent.memory_backend.create_memory_backend", return_value=backend),
            patch.object(memory_jobs, "MEMORY_JOBS", mgr),
            patch("nova.agent.memory_snapshot.refresh_memory_snapshot_silently") as refresh,
        ):
            result = asyncio.run(processor.run(
                messages,
                session_key="session_1",
                force=True,
                sync_mem0=True,
            ))

        self.assertFalse(result)
        self.assertEqual(mgr.list(), [])
        self.assertEqual(store.cursor, 0)
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
