from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeMemoryStore:
    def __init__(self, content: str = ""):
        self.content = content
        self.appended: list[str] = []

    def read_memory(self) -> str:
        return self.content

    def append_history(self, content: str, *, max_chars: int | None = None) -> int:
        self.appended.append(content)
        return len(self.appended)

    def get_last_dream_cursor(self) -> int:
        return 0

    def read_unprocessed_history(self, since_cursor: int) -> list[dict]:
        return []


class MemoryBackendTests(unittest.TestCase):
    def test_markdown_backend_exposes_memory_context_and_search_hit(self) -> None:
        from nova.agent.memory_backend import (
            MarkdownMemoryBackend,
            build_memory_scope,
        )

        backend = MarkdownMemoryBackend(FakeMemoryStore("- User prefers concise replies."))
        scope = build_memory_scope(scope="project", session_key="session_1")

        self.assertEqual(
            backend.get_memory_context(),
            "## Long-term Memory\n\n- User prefers concise replies.",
        )
        hits = backend.search("concise", scope, limit=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].id, "markdown:MEMORY.md")
        self.assertEqual(hits[0].source, "markdown")
        self.assertEqual(hits[0].metadata["scope"], "project")

    def test_memory_scope_uses_stable_workspace_hash(self) -> None:
        from nova.agent.memory_backend import build_memory_scope

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            first = build_memory_scope(
                workspace=workspace,
                session_key="session_1",
                scope="session",
                user_id="tester",
            )
            second = build_memory_scope(
                workspace=workspace,
                session_key="session_1",
                scope="session",
                user_id="tester",
            )

        self.assertEqual(first.workspace_id, second.workspace_id)
        self.assertEqual(len(first.workspace_id), 16)
        self.assertEqual(first.user_id, "tester")
        self.assertEqual(first.scope, "session")
        self.assertEqual(first.session_key, "session_1")


class MemoryStoreBaselineTests(unittest.TestCase):
    def test_history_append_read_and_cursor_are_preserved(self) -> None:
        from nova.agent import memory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / ".nova"
            memory_dir = runtime / "memory"

            with (
                patch.object(memory, "RUNTIME_DIR", runtime),
                patch.object(memory, "MEMORY_DIR", memory_dir),
                patch.object(memory, "MEMORY_FILE", memory_dir / "MEMORY.md"),
                patch.object(memory, "HISTORY_FILE", memory_dir / "history.jsonl"),
                patch.object(memory, "CURSOR_FILE", memory_dir / ".cursor"),
                patch.object(memory, "DREAM_CURSOR_FILE", memory_dir / ".dream_cursor"),
                patch.object(memory, "SOUL_MD_PATH", runtime / "SOUL.md"),
                patch.object(memory, "USER_MD_PATH", runtime / "USER.md"),
            ):
                store = memory.MemoryStore(root)

                first = store.append_history("User: remember alpha")
                second = store.append_history("Nova: noted beta")
                entries = store.read_unprocessed_history(0)

                self.assertEqual(first, 1)
                self.assertEqual(second, 2)
                self.assertEqual([entry["cursor"] for entry in entries], [1, 2])
                self.assertIn("remember alpha", entries[0]["content"])

                store.set_last_dream_cursor(2)
                self.assertEqual(store.get_last_dream_cursor(), 2)


class SystemPromptMemoryTests(unittest.TestCase):
    def test_system_prompt_includes_markdown_backend_memory(self) -> None:
        from nova.agent import context, memory_backend

        fake_store = FakeMemoryStore("- Stable project memory.")
        backend = memory_backend.MarkdownMemoryBackend(fake_store)

        with (
            patch.object(memory_backend, "_MEMORY_BACKEND", backend),
            patch.object(context, "BOOTSTRAP_FILES", []),
        ):
            prompt = context.build_system_prompt(skills_descriptions="(no skills)")

        self.assertIn("## Long-term Memory", prompt)
        self.assertIn("- Stable project memory.", prompt)

    def test_system_prompt_adds_mem0_tool_guidance_when_mem0_enabled(self) -> None:
        from nova.agent import context, memory_backend

        fake_store = FakeMemoryStore("")
        backend = memory_backend.MarkdownMemoryBackend(fake_store, backend_name="mem0")

        with (
            patch.object(memory_backend, "_MEMORY_BACKEND", backend),
            patch.object(context, "BOOTSTRAP_FILES", []),
            patch.object(context, "NOVA_MEMORY_BACKEND", "mem0"),
        ):
            prompt = context.build_system_prompt(skills_descriptions="(no skills)")

        self.assertIn("## Memory Tools", prompt)
        self.assertIn("must use the dedicated Mem0 memory tools", prompt)
        self.assertIn("Only if a Mem0 tool clearly times out or returns an error", prompt)
        self.assertIn(".nova/memory/MEMORY.md", prompt)
        self.assertIn("get_memories", prompt)
        self.assertIn("search_memories", prompt)


if __name__ == "__main__":
    unittest.main()
