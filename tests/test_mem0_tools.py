from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeMem0Client:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._memory = {
            "id": "mem-1",
            "memory": "User prefers concise Chinese replies.",
            "user_id": "tester",
            "agent_id": "nova",
            "run_id": "session-123",
            "metadata": {
                "scope": "session",
                "workspace_id": "workspace-hash",
                "project_name": "workbot",
                "session_key": "session-123",
                "app_id": "nova-ai",
            },
        }
        self._project_memories = [
            {
                "id": "mem-p1",
                "memory": "Project memory 1",
                "user_id": "tester",
                "agent_id": "nova",
                "metadata": {
                    "scope": "project",
                    "workspace_id": "workspace-hash",
                    "project_name": "workbot",
                    "app_id": "nova-ai",
                },
            },
            {
                "id": "mem-p2",
                "memory": "Project memory 2",
                "user_id": "tester",
                "agent_id": "nova",
                "metadata": {
                    "scope": "project",
                    "workspace_id": "workspace-hash",
                    "project_name": "workbot",
                    "app_id": "nova-ai",
                },
            },
            {
                "id": "mem-p3",
                "memory": "Project memory 3",
                "user_id": "tester",
                "agent_id": "nova",
                "metadata": {
                    "scope": "project",
                    "workspace_id": "workspace-hash",
                    "project_name": "workbot",
                    "app_id": "nova-ai",
                },
            },
        ]
        self._user_memory = {
            "id": "mem-u1",
            "memory": "User memory 1",
            "user_id": "tester",
            "agent_id": "nova",
            "metadata": {
                "scope": "user",
                "app_id": "nova-ai",
            },
        }

    def search(self, *args, **kwargs):
        self.calls.append(("search", args, kwargs))
        return {"results": [self._memory]}

    def get_all(self, *args, **kwargs):
        self.calls.append(("get_all", args, kwargs))
        filters = kwargs.get("filters") or {}
        scope = filters.get("scope")
        if scope == "project":
            return {"results": list(self._project_memories)}
        if scope == "session":
            return {"results": [self._memory]}
        return {"results": [self._user_memory, *self._project_memories, self._memory]}

    def get(self, memory_id):
        self.calls.append(("get", (memory_id,), {}))
        if self._memory.get("deleted"):
            return None
        if memory_id == self._memory["id"]:
            return self._memory
        return None

    def delete(self, memory_id):
        self.calls.append(("delete", (memory_id,), {}))
        if memory_id == self._memory["id"]:
            self._memory = dict(self._memory)
            self._memory["deleted"] = True
        return {"message": "Memory deleted successfully!"}


class LimitOnlyMem0Client(FakeMem0Client):
    def search(self, *args, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("Memory.search() got an unexpected keyword argument 'top_k'")
        self.calls.append(("search", args, kwargs))
        return {"results": [self._memory]}

    def get_all(self, *args, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("Memory.get_all() got an unexpected keyword argument 'top_k'")
        return super().get_all(*args, **kwargs)


class FakeMem0Backend:
    def __init__(self, client=None, error=None):
        self._client = client or FakeMem0Client()
        self._error = error
        self.snapshot_refreshes = 0

    def _get_client(self):
        return None if self._error else self._client

    def status(self):
        return type("Status", (), {"last_error": self._error})()

    def delete(self, memory_id):
        client = self._get_client()
        if client is None:
            return type("Result", (), {"ok": False, "error": self._error or "backend unavailable"})()
        client.delete(memory_id)
        if client.get(memory_id) is not None:
            return type("Result", (), {"ok": False, "error": "memory still exists after delete confirmation"})()
        return type("Result", (), {"ok": True, "error": None})()


class Mem0ToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeMem0Client()
        self.backend = FakeMem0Backend(client=self.client)

    def _scope(self, scope="project", session_key="session-123"):
        from nova.agent.memory_backend import MemoryScope

        return MemoryScope(
            user_id="tester",
            session_key=session_key,
            workspace_id="workspace-hash",
            project_name="workbot",
            scope=scope,
        )

    def test_search_memories_uses_scope_filters(self) -> None:
        from nova.tools.builtin.mem0_tools import SearchMemoriesTool

        tool = SearchMemoriesTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(query="concise replies", scope="session", limit=3))

        self.assertTrue(result["ok"])
        self.assertEqual(result["scope"], "session")
        name, args, kwargs = self.client.calls[-1]
        self.assertEqual(name, "search")
        self.assertEqual(args, ("concise replies",))
        self.assertEqual(kwargs["top_k"], 3)
        self.assertEqual(kwargs["filters"]["user_id"], "tester")
        self.assertEqual(kwargs["filters"]["run_id"], "session-123")
        self.assertEqual(kwargs["filters"]["scope"], "session")
        self.assertEqual(result["results"][0]["memory_id"], "mem-1")

    def test_get_memories_applies_pagination(self) -> None:
        from nova.tools.builtin.mem0_tools import GetMemoriesTool

        tool = GetMemoriesTool()

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(scope="project", page=2, page_size=1))

        self.assertTrue(result["ok"])
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["page_size"], 1)
        self.assertEqual(result["count"], 1)
        self.assertTrue(result["has_more"])
        self.assertEqual(result["results"][0]["id"], "mem-p2")
        self.assertEqual(result["results"][0]["memory_id"], "mem-p2")
        name, _args, kwargs = self.client.calls[-1]
        self.assertEqual(name, "get_all")
        self.assertEqual(kwargs["top_k"], 3)
        self.assertEqual(kwargs["filters"]["workspace_id"], "workspace-hash")

    def test_get_memories_defaults_to_current_project_plus_session(self) -> None:
        from nova.tools.builtin.mem0_tools import GetMemoriesTool

        tool = GetMemoriesTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(page=1, page_size=10))

        self.assertTrue(result["ok"])
        self.assertEqual(result["scope"], "current")
        self.assertEqual(result["count"], 4)
        self.assertEqual([item["memory_id"] for item in result["results"]], ["mem-p1", "mem-p2", "mem-p3", "mem-1"])
        self.assertEqual(self.client.calls[0][2]["filters"]["scope"], "project")
        self.assertEqual(self.client.calls[1][2]["filters"]["scope"], "session")

    def test_get_memories_any_groups_results_by_scope(self) -> None:
        from nova.tools.builtin.mem0_tools import GetMemoriesTool

        tool = GetMemoriesTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(scope="any", page=1, page_size=10))

        self.assertTrue(result["ok"])
        self.assertEqual(result["scope"], "any")
        self.assertEqual(list(result["grouped_results"].keys()), ["user", "project", "session"])
        self.assertEqual(result["grouped_results"]["session"][0]["memory_id"], "mem-1")

    def test_memory_tools_support_limit_only_sdk_methods(self) -> None:
        from nova.tools.builtin.mem0_tools import GetMemoriesTool, SearchMemoriesTool

        client = LimitOnlyMem0Client()
        backend = FakeMem0Backend(client=client)
        search_tool = SearchMemoriesTool()
        list_tool = GetMemoriesTool()

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            search_result = json.loads(search_tool.execute(query="concise", scope="project", limit=4))
            list_result = json.loads(list_tool.execute(scope="project", page=1, page_size=1))

        self.assertTrue(search_result["ok"])
        self.assertTrue(list_result["ok"])
        self.assertEqual(client.calls[0][2]["limit"], 4)
        self.assertNotIn("top_k", client.calls[0][2])
        self.assertEqual(client.calls[1][2]["limit"], 2)
        self.assertNotIn("top_k", client.calls[1][2])

    def test_get_memory_checks_scope(self) -> None:
        from nova.tools.builtin.mem0_tools import GetMemoryTool

        tool = GetMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(memory_id="mem-1", scope="session"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["memory_id"], "mem-1")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(memory_id="mem-1", scope="project"))

        self.assertFalse(result["ok"])
        self.assertIn("outside the requested scope", result["error"])

    def test_delete_memory_tool_deletes_when_scope_matches(self) -> None:
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool

        tool = DeleteMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            result = json.loads(tool.execute(memory_id="mem-1", scope="session"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["memory_id"], "mem-1")
        self.assertEqual(result["deleted_memory"]["memory_id"], "mem-1")
        self.assertEqual(result["deleted_memory"]["memory"], "User prefers concise Chinese replies.")
        self.assertEqual(self.client.calls[-2][0], "delete")
        self.assertEqual(self.client.calls[-1], ("get", ("mem-1",), {}))

    def test_delete_memory_tool_refreshes_snapshot_after_success(self) -> None:
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool

        tool = DeleteMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
            patch("nova.agent.memory_snapshot.refresh_memory_snapshot_silently") as refresh,
        ):
            result = json.loads(tool.execute(memory_id="mem-1", scope="session"))

        self.assertTrue(result["ok"])
        refresh.assert_called_once()

    def test_delete_memory_tool_records_telemetry(self) -> None:
        from nova.agent.memory_telemetry import MEMORY_TELEMETRY
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool

        tool = DeleteMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
            patch(
                "nova.agent.memory_snapshot.refresh_memory_snapshot_silently",
                return_value=type("Snapshot", (), {"ok": True, "error": None})(),
            ),
        ):
            result = json.loads(tool.execute(memory_id="mem-1", scope="session"))

        self.assertTrue(result["ok"])
        telemetry = MEMORY_TELEMETRY.snapshot()
        self.assertEqual(telemetry.last_delete_scope, "session")
        self.assertEqual(telemetry.last_delete_memory_id, "mem-1")
        self.assertTrue(telemetry.last_snapshot_refresh_ok)

    def test_delete_memory_permission_preview_includes_memory_text(self) -> None:
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool

        tool = DeleteMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        with (
            patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
            patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
        ):
            preview = tool.permission_preview({"memory_id": "mem-1", "scope": "session"})

        self.assertEqual(preview["memory_id"], "mem-1")
        self.assertEqual(preview["scope"], "session")
        self.assertEqual(preview["actual_scope"], "session")
        self.assertEqual(preview["memory_preview"], "User prefers concise Chinese replies.")

    def test_delete_memory_requires_permission_via_orchestration(self) -> None:
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool
        from nova.tools.orchestration import execute_tool_batches
        from nova.tools.registry import register_tool

        tool = DeleteMemoryTool()
        register_tool(tool)

        async def _run():
            with patch("nova.tools.orchestration.PERMISSIONS.authorize") as authorize:
                authorize.return_value = type(
                    "Decision",
                    (),
                    {
                        "behavior": "deny",
                        "message": "Permission denied for tool 'delete_memory'.",
                        "updated_params": None,
                    },
                )()
                return await execute_tool_batches([
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "delete_memory",
                            "arguments": json.dumps({"memory_id": "mem-1"}, ensure_ascii=False),
                        },
                    }
                ])

        import asyncio

        results = asyncio.run(_run())
        self.assertEqual(len(results), 1)
        self.assertIn("Permission denied", str(results[0]["output"]))

    def test_delete_memory_permission_prompt_shows_memory_preview(self) -> None:
        from nova.permissions.manager import PermissionManager
        from nova.tools.builtin.mem0_tools import DeleteMemoryTool

        captured: list[dict] = []

        async def prompt_handler(request):
            captured.append(request)
            return {"action": "deny"}

        tool = DeleteMemoryTool()
        tool.set_runtime_context(session_key="session-123")

        async def _run():
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch("nova.tools.builtin.mem0_tools.create_memory_backend", return_value=self.backend),
                patch("nova.tools.builtin.mem0_tools.build_memory_scope", side_effect=lambda **kwargs: self._scope(kwargs.get("scope", "project"), kwargs.get("session_key"))),
            ):
                manager = PermissionManager(Path(tmp) / "permissions.json")
                manager.set_prompt_handler(prompt_handler)
                return await manager.authorize(
                    "delete_memory",
                    {"memory_id": "mem-1", "scope": "session"},
                    tool,
                )

        import asyncio

        decision = asyncio.run(_run())
        self.assertEqual(decision.behavior, "deny")
        self.assertEqual(len(captured), 1)
        self.assertIn("memory_id=mem-1", captured[0]["message"])
        self.assertIn("scope=session", captured[0]["message"])
        self.assertIn("memory=User prefers concise Chinese replies.", captured[0]["message"])


if __name__ == "__main__":
    unittest.main()
