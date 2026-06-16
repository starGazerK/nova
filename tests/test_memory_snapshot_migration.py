from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path


class FakeGit:
    def __init__(self):
        self.commits: list[str] = []

    def auto_commit(self, message: str) -> str:
        self.commits.append(message)
        return "abc12345"


class FakeSnapshotStore:
    def __init__(self, root: Path):
        self.workspace = root
        self.memory_dir = root / ".nova" / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.user = ""
        self.soul = ""
        self.memory = ""
        self.git = FakeGit()
        self.git_initialized = False

    def ensure_git_initialized(self) -> bool:
        self.git_initialized = True
        return True

    def read_user(self) -> str:
        return self.user or "(empty)"

    def read_soul(self) -> str:
        return self.soul or "(empty)"

    def read_memory(self) -> str:
        return self.memory or "(empty)"


class FakeMemoryBackend:
    def __init__(self, *, healthy: bool = True, ok: bool = True):
        from nova.agent.memory_backend import MemoryBackendStatus, MemoryWriteResult

        self.healthy = healthy
        self.ok = ok
        self.get_all_calls = []
        self.add_calls = []
        self._status_cls = MemoryBackendStatus
        self._write_cls = MemoryWriteResult

    def status(self):
        return self._status_cls(
            enabled=True,
            backend="mem0",
            healthy=self.healthy,
            pending_writes=0,
            last_error=None if self.healthy else "qdrant unavailable",
            last_write_at=None,
        )

    def get_all(self, scope, *, limit=None):
        from nova.agent.memory_backend import MemoryHit

        self.get_all_calls.append((scope, limit))
        if scope.scope == "user":
            return [MemoryHit("u1", "User prefers Chinese replies.", 0.9, {}, "mem0")]
        if scope.scope == "project":
            return [MemoryHit("p1", "Project is Nova.", 0.8, {}, "mem0")]
        if scope.scope == "session":
            return [MemoryHit("s1", "Session discussed migration.", 0.7, {}, "mem0")]
        return []

    def add_messages(self, messages, scope, metadata):
        self.add_calls.append((messages, scope, metadata))
        if self.ok:
            return self._write_cls(ok=True, ids=[metadata["legacy_file"]])
        return self._write_cls(ok=False, ids=[], error="write failed")


class FailingSnapshotBackend(FakeMemoryBackend):
    def get_all(self, scope, *, limit=None):
        raise RuntimeError("snapshot query failed")


class MemorySnapshotTests(unittest.TestCase):
    def test_render_memory_snapshot_sections_and_limit(self) -> None:
        from nova.agent.memory_backend import MemoryHit
        from nova.agent.memory_snapshot import render_memory_snapshot

        content = render_memory_snapshot(
            user_hits=[MemoryHit("u1", " User prefers Chinese replies. ", 0.9, {}, "mem0")],
            project_hits=[MemoryHit("p1", "Project is Nova.", 0.8, {}, "mem0")],
            session_hits=[MemoryHit("s1", "Session discussed migration.", 0.7, {}, "mem0")],
            generated_at=datetime(2026, 6, 8, 12, 0, 0),
            limit=2,
        )

        self.assertIn("# Memory Snapshot", content)
        self.assertIn("Generated from Mem0 at 2026-06-08 12:00:00.", content)
        self.assertIn("## User Memory", content)
        self.assertIn("- User prefers Chinese replies.", content)
        self.assertIn("- Project is Nova.", content)
        self.assertNotIn("Session discussed migration.", content)

    def test_export_memory_snapshot_writes_file_and_commits(self) -> None:
        from nova.agent.memory_snapshot import export_memory_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeSnapshotStore(Path(tmp))
            backend = FakeMemoryBackend()

            result = export_memory_snapshot(
                backend=backend,  # type: ignore[arg-type]
                store=store,
                session_key="session_1",
                limit=10,
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.count, 3)
            self.assertTrue(store.git_initialized)
            self.assertEqual(store.git.commits, ["memory: export mem0 snapshot"])
            written = store.memory_file.read_text(encoding="utf-8")
            self.assertIn("## Recent Session Memory", written)
            self.assertIn("- Session discussed migration.", written)

    def test_export_memory_snapshot_surfaces_backend_query_failure(self) -> None:
        from nova.agent.memory_snapshot import export_memory_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeSnapshotStore(Path(tmp))
            backend = FailingSnapshotBackend()

            result = export_memory_snapshot(
                backend=backend,  # type: ignore[arg-type]
                store=store,
                session_key="session_1",
                limit=10,
            )

            self.assertFalse(result.ok)
            self.assertIn("snapshot query failed", result.error or "")


class MemoryMigrationTests(unittest.TestCase):
    def test_legacy_migration_writes_batches_marker_and_snapshot(self) -> None:
        from nova.agent.memory_migration import run_legacy_memory_migration

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeSnapshotStore(Path(tmp))
            store.user = "- Language: Chinese"
            store.soul = "- Tone: concise"
            store.memory = "- Project uses Mem0"
            backend = FakeMemoryBackend()

            result = run_legacy_memory_migration(
                backend=backend,  # type: ignore[arg-type]
                store=store,
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.migrated, 3)
            self.assertFalse(result.skipped)
            self.assertEqual(len(backend.add_calls), 3)
            self.assertTrue((store.memory_dir / ".mem0_migration").exists())
            self.assertIn("legacy_markdown_migration", backend.add_calls[0][2]["source"])
            self.assertIn("# Memory Snapshot", store.memory_file.read_text(encoding="utf-8"))

            skipped = run_legacy_memory_migration(
                backend=backend,  # type: ignore[arg-type]
                store=store,
            )
            self.assertTrue(skipped.ok)
            self.assertTrue(skipped.skipped)

    def test_legacy_migration_failure_does_not_write_marker(self) -> None:
        from nova.agent.memory_migration import run_legacy_memory_migration

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeSnapshotStore(Path(tmp))
            store.memory = "- Project uses Mem0"
            backend = FakeMemoryBackend(ok=False)

            result = run_legacy_memory_migration(
                backend=backend,  # type: ignore[arg-type]
                store=store,
            )

            self.assertFalse(result.ok)
            self.assertIn("write failed", result.error or "")
            self.assertFalse((store.memory_dir / ".mem0_migration").exists())


if __name__ == "__main__":
    unittest.main()
