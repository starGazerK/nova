from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class FakeMem0Client:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.memories: dict[str, dict] = {
            "mem-1": {
                "id": "mem-1",
                "memory": "User prefers focused implementation steps.",
                "score": 0.91,
                "metadata": {"scope": "project"},
            },
            "mem-3": {"id": "mem-3", "memory": "Project uses Nova."},
        }

    def search(self, *args, **kwargs):
        self.calls.append(("search", args, kwargs))
        return {"results": [self.memories["mem-1"]]}

    def add(self, *args, **kwargs):
        self.calls.append(("add", args, kwargs))
        return {"results": [{"id": "mem-2", "event": "ADD"}]}

    def get_all(self, *args, **kwargs):
        self.calls.append(("get_all", args, kwargs))
        return {"results": [self.memories["mem-3"]]}

    def get(self, memory_id):
        self.calls.append(("get", (memory_id,), {}))
        return self.memories.get(memory_id)

    def update(self, *args, **kwargs):
        self.calls.append(("update", args, kwargs))
        return {"message": "updated"}

    def delete(self, *args, **kwargs):
        self.calls.append(("delete", args, kwargs))
        if args:
            self.memories.pop(args[0], None)
        return {"message": "deleted"}


class LimitOnlyMem0Client(FakeMem0Client):
    def search(self, *args, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("Memory.search() got an unexpected keyword argument 'top_k'")
        self.calls.append(("search", args, kwargs))
        return {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "Search works with limit.",
                    "metadata": {"scope": "project"},
                }
            ]
        }

    def get_all(self, *args, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("Memory.get_all() got an unexpected keyword argument 'top_k'")
        self.calls.append(("get_all", args, kwargs))
        return {"results": [{"id": "mem-3", "memory": "List works with limit."}]}


class Mem0BackendTests(unittest.TestCase):
    def _scope(self, scope: str = "session"):
        from nova.agent.memory_backend import build_memory_scope

        return build_memory_scope(
            user_id="tester",
            session_key="session_123",
            scope=scope,  # type: ignore[arg-type]
        )

    def test_build_mem0_filters_for_session_scope(self) -> None:
        from nova.agent.mem0_backend import build_mem0_filters

        filters = build_mem0_filters(self._scope("session"))

        self.assertEqual(filters["user_id"], "tester")
        self.assertEqual(filters["agent_id"], "nova")
        self.assertEqual(filters["run_id"], "session_123")
        self.assertEqual(filters["scope"], "session")
        self.assertIn("workspace_id", filters)

    def test_mem0_backend_normalizes_search_results_and_kwargs(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        client = FakeMem0Client()
        backend = Mem0MemoryBackend(client=client)
        hits = backend.search("implementation", self._scope("project"), limit=5)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].id, "mem-1")
        self.assertEqual(hits[0].text, "User prefers focused implementation steps.")
        name, args, kwargs = client.calls[-1]
        self.assertEqual(name, "search")
        self.assertEqual(args, ("implementation",))
        self.assertEqual(kwargs["top_k"], 5)
        self.assertEqual(kwargs["filters"]["scope"], "project")

    def test_mem0_backend_does_not_use_static_system_prompt_memory(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        backend = Mem0MemoryBackend(client=FakeMem0Client())

        self.assertEqual(backend.get_memory_context(), "")

    def test_mem0_backend_supports_limit_only_sdk_methods(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        client = LimitOnlyMem0Client()
        backend = Mem0MemoryBackend(client=client)

        search_hits = backend.search("implementation", self._scope("project"), limit=5)
        list_hits = backend.get_all(self._scope("project"), limit=7)

        self.assertEqual(search_hits[0].text, "Search works with limit.")
        self.assertEqual(list_hits[0].text, "List works with limit.")
        self.assertEqual(client.calls[0][2]["limit"], 5)
        self.assertNotIn("top_k", client.calls[0][2])
        self.assertEqual(client.calls[1][2]["limit"], 7)
        self.assertNotIn("top_k", client.calls[1][2])

    def test_mem0_backend_add_update_delete_wrap_results(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        client = FakeMem0Client()
        backend = Mem0MemoryBackend(client=client)
        scope = self._scope("session")

        add_result = backend.add_messages(
            [{"role": "user", "content": "Remember this."}],
            scope,
            {"source": "unit_test", "mem0_infer": False},
        )
        update_result = backend.update("mem-2", "Updated text")
        delete_result = backend.delete("mem-2")

        self.assertTrue(add_result.ok)
        self.assertEqual(add_result.ids, ["mem-2"])
        add_call = client.calls[0]
        self.assertEqual(add_call[0], "add")
        self.assertEqual(add_call[2]["user_id"], "tester")
        self.assertEqual(add_call[2]["run_id"], "session_123")
        self.assertFalse(add_call[2]["infer"])
        self.assertEqual(add_call[2]["metadata"]["source"], "unit_test")
        self.assertEqual(add_call[2]["metadata"]["app_id"], "nova-ai")
        self.assertEqual(add_call[2]["metadata"]["workspace_id"], scope.workspace_id)
        self.assertNotIn("mem0_infer", add_call[2]["metadata"])

        self.assertTrue(update_result.ok)
        self.assertTrue(delete_result.ok)
        self.assertEqual(client.calls[1], ("update", ("mem-2",), {"data": "Updated text"}))
        self.assertEqual(client.calls[2], ("delete", ("mem-2",), {}))
        self.assertEqual(client.calls[3], ("get", ("mem-2",), {}))

    def test_mem0_backend_delete_reports_error_when_memory_still_exists(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        class StickyDeleteClient(FakeMem0Client):
            def delete(self, *args, **kwargs):
                self.calls.append(("delete", args, kwargs))
                return {"message": "deleted"}

        client = StickyDeleteClient()
        backend = Mem0MemoryBackend(client=client)

        result = backend.delete("mem-1")

        self.assertFalse(result.ok)
        self.assertIn("still exists", result.error or "")

    def test_mem0_backend_reports_unhealthy_on_factory_failure(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        def failing_factory(config):
            raise RuntimeError("qdrant unavailable")

        backend = Mem0MemoryBackend(memory_factory=failing_factory)

        self.assertEqual(backend.search("anything", self._scope("project"), limit=3), [])
        status = backend.status()
        self.assertFalse(status.healthy)
        self.assertIn("qdrant unavailable", status.last_error or "")

    def test_prepare_environment_keeps_mem0_runtime_local(self) -> None:
        from nova.agent.mem0_backend import _prepare_mem0_environment
        from nova.config import MEM0_RUNTIME_DIR

        with patch.dict(os.environ, {}, clear=True):
            _prepare_mem0_environment()
            self.assertEqual(os.environ["MEM0_DIR"], str(MEM0_RUNTIME_DIR))
            self.assertEqual(os.environ["MEM0_TELEMETRY"], "false")
            self.assertIn("localhost", os.environ["NO_PROXY"])
            self.assertIn("127.0.0.1", os.environ["NO_PROXY"])

    def test_build_mem0_config_accepts_explicit_provider_settings(self) -> None:
        from nova.agent import mem0_backend

        with (
            patch.object(mem0_backend, "MEM0_LLM_PROVIDER", "openai"),
            patch.object(mem0_backend, "MEM0_LLM_MODEL", "gpt-4o-mini"),
            patch.object(mem0_backend, "MEM0_LLM_API_KEY", "llm-key"),
            patch.object(mem0_backend, "MEM0_LLM_BASE_URL", "https://llm.example/v1"),
            patch.object(mem0_backend, "MEM0_EMBEDDER_PROVIDER", "openai"),
            patch.object(mem0_backend, "MEM0_EMBEDDER_MODEL", "text-embedding-3-small"),
            patch.object(mem0_backend, "MEM0_EMBEDDER_API_KEY", "embed-key"),
            patch.object(mem0_backend, "MEM0_EMBEDDER_BASE_URL", "https://embed.example/v1"),
            patch.object(mem0_backend, "MEM0_EMBEDDER_DIMS", "1536"),
        ):
            config = mem0_backend.build_mem0_config()

        self.assertEqual(config["llm"]["provider"], "openai")
        self.assertEqual(config["llm"]["config"]["model"], "gpt-4o-mini")
        self.assertEqual(config["llm"]["config"]["api_key"], "llm-key")
        self.assertEqual(config["llm"]["config"]["openai_base_url"], "https://llm.example/v1")
        self.assertEqual(config["embedder"]["provider"], "openai")
        self.assertEqual(config["embedder"]["config"]["model"], "text-embedding-3-small")
        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 1536)
        self.assertEqual(config["vector_store"]["config"]["embedding_model_dims"], 1536)

    def test_get_client_recreates_empty_collection_when_dims_mismatch(self) -> None:
        from nova.agent import mem0_backend
        from nova.agent.mem0_backend import Mem0MemoryBackend

        factory_calls: list[dict] = []

        def factory(config):
            factory_calls.append(config)
            return FakeMem0Client()

        info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=1536))
            ),
            points_count=0,
        )

        class FakeQdrantClient:
            instances: list["FakeQdrantClient"] = []

            def __init__(self, *, host: str, port: int):
                self.host = host
                self.port = port
                self.deleted: list[str] = []
                FakeQdrantClient.instances.append(self)

            def collection_exists(self, name: str) -> bool:
                return True

            def get_collection(self, name: str):
                return info

            def delete_collection(self, name: str) -> None:
                self.deleted.append(name)

        with (
            patch.object(mem0_backend, "MEM0_EMBEDDER_DIMS", "1024"),
            patch.object(mem0_backend, "MEM0_COLLECTION", "nova_memories"),
            patch.object(mem0_backend, "MEM0_QDRANT_HOST", "localhost"),
            patch.object(mem0_backend, "MEM0_QDRANT_PORT", 6335),
            patch("qdrant_client.QdrantClient", FakeQdrantClient),
        ):
            backend = Mem0MemoryBackend(memory_factory=factory)
            client = backend._get_client()

        self.assertIsInstance(client, FakeMem0Client)
        self.assertEqual(FakeQdrantClient.instances[0].deleted, ["nova_memories"])
        self.assertEqual(
            factory_calls[0]["vector_store"]["config"]["embedding_model_dims"],
            1024,
        )

    def test_get_client_fails_for_non_empty_collection_dim_mismatch(self) -> None:
        from nova.agent import mem0_backend
        from nova.agent.mem0_backend import Mem0MemoryBackend

        info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=1536))
            ),
            points_count=3,
        )

        class FakeQdrantClient:
            def __init__(self, *, host: str, port: int):
                self.host = host
                self.port = port

            def collection_exists(self, name: str) -> bool:
                return True

            def get_collection(self, name: str):
                return info

        with (
            patch.object(mem0_backend, "MEM0_EMBEDDER_DIMS", "1024"),
            patch.object(mem0_backend, "MEM0_COLLECTION", "nova_memories"),
            patch.object(mem0_backend, "MEM0_QDRANT_HOST", "localhost"),
            patch.object(mem0_backend, "MEM0_QDRANT_PORT", 6335),
            patch("qdrant_client.QdrantClient", FakeQdrantClient),
        ):
            backend = Mem0MemoryBackend(memory_factory=lambda config: FakeMem0Client())
            client = backend._get_client()

        self.assertIsNone(client)
        self.assertIn("dimension mismatch", backend._last_error or "")
        self.assertIn("configured 1024", backend._last_error or "")
        self.assertIn("uses 1536", backend._last_error or "")


if __name__ == "__main__":
    unittest.main()
