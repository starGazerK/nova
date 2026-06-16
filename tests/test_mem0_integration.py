from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


@unittest.skipUnless(
    os.getenv("NOVA_TEST_MEM0") == "1",
    "set NOVA_TEST_MEM0=1 to run live Mem0/Qdrant integration tests",
)
class Mem0IntegrationTests(unittest.TestCase):
    def test_mem0_empty_collection_with_new_dims_is_recreated(self) -> None:
        from qdrant_client import QdrantClient

        collection_name = f"nova_memories_dims_{int(time.time())}"
        client = QdrantClient(host="localhost", port=6335)
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(collection_name=collection_name, vectors_config={"size": 1536, "distance": "Cosine"})
        info_before = client.get_collection(collection_name)
        self.assertEqual(info_before.config.params.vectors.size, 1536)
        self.assertEqual(info_before.points_count, 0)

        from nova.agent import mem0_backend
        from nova.agent.mem0_backend import Mem0MemoryBackend

        with (
            mock.patch.object(mem0_backend, "MEM0_COLLECTION", collection_name),
            mock.patch.object(mem0_backend, "MEM0_EMBEDDER_DIMS", "1024"),
        ):
            config = mem0_backend.build_mem0_config()
            backend = Mem0MemoryBackend(memory_factory=lambda _: object())
            backend._ensure_qdrant_collection_ready()
            recreated = QdrantClient(host="localhost", port=6335)
            recreated.create_collection(collection_name=collection_name, vectors_config={"size": config["vector_store"]["config"]["embedding_model_dims"], "distance": "Cosine"})

        info_after = client.get_collection(collection_name)
        self.assertEqual(info_after.config.params.vectors.size, 1024)

    def test_mem0_add_search_and_export_against_qdrant(self) -> None:
        from nova.agent.mem0_backend import Mem0MemoryBackend
        from nova.agent.memory_backend import build_memory_scope
        from nova.agent.memory_snapshot import export_memory_snapshot

        class FakeGit:
            def auto_commit(self, message: str):
                return "integration"

        class Store:
            def __init__(self, root: Path):
                self.workspace = root
                self.memory_dir = root / ".nova" / "memory"
                self.memory_file = self.memory_dir / "MEMORY.md"
                self.git = FakeGit()

            def ensure_git_initialized(self):
                return True

        unique_text = f"Nova integration memory {int(time.time())}"
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            backend = Mem0MemoryBackend(backend_name="mem0")
            scope = build_memory_scope(scope="project", workspace=store.workspace)

            status = backend.status()
            self.assertTrue(status.healthy, status.last_error)
            result = backend.add_messages(
                [{"role": "user", "content": unique_text}],
                scope,
                {"source": "integration_test"},
            )
            self.assertTrue(result.ok, result.error)

            hits = backend.search(unique_text, scope, limit=3)
            self.assertTrue(
                any(unique_text.lower() in hit.text.lower() for hit in hits),
                hits,
            )
            snapshot = export_memory_snapshot(
                backend=backend,
                store=store,
                limit=10,
            )
            self.assertTrue(snapshot.ok, snapshot.error)
            self.assertIn(unique_text, store.memory_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
