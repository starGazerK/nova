"""
nova/agent/mem0_backend.py - Mem0-backed memory adapter.

This module is imported only when the memory backend flag selects Mem0, so the
default Markdown path remains free of Mem0 import side effects.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from nova.agent.memory_backend import (
    MemoryBackendStatus,
    MemoryHit,
    MemoryScope,
    MemoryWriteResult,
)
from nova.config import (
    MEM0_COLLECTION,
    MEM0_EMBEDDER_API_KEY,
    MEM0_EMBEDDER_BASE_URL,
    MEM0_EMBEDDER_DIMS,
    MEM0_EMBEDDER_MODEL,
    MEM0_EMBEDDER_PROVIDER,
    MEM0_ENABLE_GRAPH,
    MEM0_LLM_API_KEY,
    MEM0_LLM_BASE_URL,
    MEM0_LLM_MODEL,
    MEM0_LLM_PROVIDER,
    MEM0_QDRANT_HOST,
    MEM0_QDRANT_PORT,
    MEM0_RUNTIME_DIR,
)

_AGENT_ID = "nova"
_APP_ID = "nova-ai"


def _parse_embedding_dims(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dims = int(value)
    except (TypeError, ValueError):
        return None
    return dims if dims > 0 else None


def build_mem0_config() -> dict[str, Any]:
    """Build the local Mem0 OSS config for Nova's Qdrant backend."""
    embedding_dims = _parse_embedding_dims(MEM0_EMBEDDER_DIMS)
    vector_store_config: dict[str, Any] = {
        "collection_name": MEM0_COLLECTION,
        "host": MEM0_QDRANT_HOST,
        "port": MEM0_QDRANT_PORT,
    }
    if embedding_dims is not None:
        vector_store_config["embedding_model_dims"] = embedding_dims
    config: dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": vector_store_config,
        },
        "history_db_path": str(MEM0_RUNTIME_DIR / "history.db"),
    }
    llm = _build_provider_config(
        provider=MEM0_LLM_PROVIDER,
        model=MEM0_LLM_MODEL,
        api_key=MEM0_LLM_API_KEY,
        base_url=MEM0_LLM_BASE_URL,
        base_url_key="openai_base_url",
    )
    if llm:
        config["llm"] = llm
    embedder = _build_provider_config(
        provider=MEM0_EMBEDDER_PROVIDER,
        model=MEM0_EMBEDDER_MODEL,
        api_key=MEM0_EMBEDDER_API_KEY,
        base_url=MEM0_EMBEDDER_BASE_URL,
        base_url_key="openai_base_url",
        embedding_dims=MEM0_EMBEDDER_DIMS,
    )
    if embedder:
        config["embedder"] = embedder
    return config


def _build_provider_config(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    base_url_key: str,
    embedding_dims: str | None = None,
) -> dict[str, Any] | None:
    if not provider:
        return None
    provider_config: dict[str, Any] = {}
    if model:
        provider_config["model"] = model
    if api_key:
        provider_config["api_key"] = api_key
    if base_url:
        provider_config[base_url_key] = base_url
    parsed_dims = _parse_embedding_dims(embedding_dims)
    if parsed_dims is not None:
        provider_config["embedding_dims"] = parsed_dims
    return {
        "provider": provider,
        "config": provider_config,
    }


def build_mem0_filters(scope: MemoryScope) -> dict[str, Any]:
    """Build filters accepted by the local Mem0 OSS SDK."""
    filters: dict[str, Any] = {
        "user_id": scope.user_id,
        "agent_id": _AGENT_ID,
        "app_id": _APP_ID,
        "scope": scope.scope,
    }
    if scope.scope == "session" and scope.session_key:
        filters["run_id"] = scope.session_key
        filters["session_key"] = scope.session_key
    if scope.scope in {"session", "project"}:
        filters["workspace_id"] = scope.workspace_id
        filters["project_name"] = scope.project_name
    return filters


def build_mem0_metadata(scope: MemoryScope, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build storage metadata for Nova-created memories."""
    merged = dict(metadata or {})
    merged.update({
        "scope": scope.scope,
        "workspace_id": scope.workspace_id,
        "project_name": scope.project_name,
        "created_by": "nova",
        "app_id": _APP_ID,
    })
    if scope.session_key:
        merged["session_key"] = scope.session_key
    return merged


def _prepare_mem0_environment() -> None:
    """Keep Mem0 runtime files inside Nova's writable runtime directory."""
    MEM0_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MEM0_DIR", str(MEM0_RUNTIME_DIR))
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    required = {"localhost", "127.0.0.1", "::1"}
    current = {item.strip() for item in no_proxy.split(",") if item.strip()}
    missing = sorted(required - current)
    if missing:
        os.environ["NO_PROXY"] = ",".join(sorted(current | required))


def _extract_ids(result: Any) -> list[str]:
    if isinstance(result, dict):
        raw_items = result.get("results") or result.get("memories") or []
    elif isinstance(result, list):
        raw_items = result
    else:
        raw_items = []
    ids: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            value = item.get("id") or item.get("memory_id")
            if value is not None:
                ids.append(str(value))
    return ids


def _normalize_mem0_results(result: Any) -> list[MemoryHit]:
    if isinstance(result, dict):
        raw_items = result.get("results") or result.get("memories") or []
    elif isinstance(result, list):
        raw_items = result
    else:
        raw_items = []

    hits: list[MemoryHit] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = item.get("memory") or item.get("text") or item.get("data") or item.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"id", "memory_id", "memory", "text", "data", "content", "score"}
            }
        hits.append(
            MemoryHit(
                id=str(item.get("id") or item.get("memory_id")) if item.get("id") or item.get("memory_id") else None,
                text=text.strip(),
                score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
                metadata=metadata,
                source="mem0",
            )
        )
    return hits


def _uses_limit_instead_of_top_k(exc: TypeError) -> bool:
    message = str(exc)
    return "top_k" in message and "unexpected keyword argument" in message


def _scope_identity_kwargs(scope: MemoryScope) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "user_id": scope.user_id,
        "agent_id": _AGENT_ID,
    }
    if scope.scope == "session" and scope.session_key:
        kwargs["run_id"] = scope.session_key
    return kwargs


def _search_mem0(
    client: Any,
    query: str,
    *,
    limit: int,
    filters: dict[str, Any],
    identity: dict[str, Any],
) -> Any:
    try:
        return client.search(query, top_k=limit, filters=filters, **identity)
    except TypeError as exc:
        if not _uses_limit_instead_of_top_k(exc):
            raise
        return client.search(query, limit=limit, filters=filters, **identity)


def _get_all_mem0(
    client: Any,
    *,
    limit: int,
    filters: dict[str, Any],
    identity: dict[str, Any],
) -> Any:
    try:
        return client.get_all(filters=filters, top_k=limit, **identity)
    except TypeError as exc:
        if not _uses_limit_instead_of_top_k(exc):
            raise
        return client.get_all(filters=filters, limit=limit, **identity)


def _get_collection_vector_size(info: Any) -> int | None:
    try:
        vectors = info.config.params.vectors
    except AttributeError:
        return None
    if isinstance(vectors, dict):
        default_vector = vectors.get("") or vectors.get("default")
        if isinstance(default_vector, dict):
            size = default_vector.get("size")
            return size if isinstance(size, int) else None
        size = vectors.get("size")
        return size if isinstance(size, int) else None
    size = getattr(vectors, "size", None)
    return size if isinstance(size, int) else None


def _get_collection_points_count(info: Any) -> int:
    for attr in ("points_count", "vectors_count", "indexed_vectors_count"):
        value = getattr(info, attr, None)
        if isinstance(value, int):
            return value
    return 0


class Mem0MemoryBackend:
    """Mem0-backed implementation of the Nova memory backend protocol."""

    def __init__(
        self,
        *,
        backend_name: str = "mem0",
        client: Any | None = None,
        memory_factory: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.backend_name = backend_name
        self._client = client
        self._memory_factory = memory_factory
        self._last_error: str | None = None
        self._last_write_at: float | None = None

    def get_memory_context(self) -> str:
        """Phase 2 wires the backend only; prompt retrieval starts in Phase 3."""
        return ""

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            _prepare_mem0_environment()
            self._ensure_qdrant_collection_ready()
            if self._memory_factory is None:
                from mem0 import Memory

                self._memory_factory = Memory.from_config
            self._client = self._memory_factory(build_mem0_config())
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            return None
        return self._client

    def _ensure_qdrant_collection_ready(self) -> None:
        expected_dims = _parse_embedding_dims(MEM0_EMBEDDER_DIMS)
        if expected_dims is None:
            return
        try:
            from qdrant_client import QdrantClient
        except Exception:
            return

        client = QdrantClient(host=MEM0_QDRANT_HOST, port=MEM0_QDRANT_PORT)
        if not client.collection_exists(MEM0_COLLECTION):
            return

        info = client.get_collection(MEM0_COLLECTION)
        actual_dims = _get_collection_vector_size(info)
        if actual_dims is None or actual_dims == expected_dims:
            return

        points_count = _get_collection_points_count(info)
        if points_count == 0:
            client.delete_collection(MEM0_COLLECTION)
            return

        raise RuntimeError(
            "Mem0 embedding dimension mismatch for collection "
            f"'{MEM0_COLLECTION}': configured {expected_dims}, existing Qdrant "
            f"collection uses {actual_dims}. The collection already contains "
            f"{points_count} vector(s), so Nova will not recreate it "
            "automatically. Use a new MEM0_COLLECTION value or clear the "
            "existing collection before retrying."
        )

    def search(self, query: str, scope: MemoryScope, *, limit: int) -> list[MemoryHit]:
        if limit <= 0:
            return []
        client = self._get_client()
        if client is None:
            return []
        try:
            result = _search_mem0(
                client,
                query,
                limit=limit,
                filters=build_mem0_filters(scope),
                identity=_scope_identity_kwargs(scope),
            )
            self._last_error = None
            return _normalize_mem0_results(result)
        except Exception as exc:
            self._last_error = str(exc)
            return []

    def add_messages(
        self,
        messages: list[dict],
        scope: MemoryScope,
        metadata: dict,
    ) -> MemoryWriteResult:
        client = self._get_client()
        if client is None:
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)
        try:
            infer = bool(metadata.get("mem0_infer", True))
            write_metadata = build_mem0_metadata(
                scope,
                {key: value for key, value in metadata.items() if key != "mem0_infer"},
            )
            result = client.add(
                messages,
                user_id=scope.user_id,
                agent_id=_AGENT_ID,
                run_id=scope.session_key if scope.scope == "session" else None,
                metadata=write_metadata,
                infer=infer,
            )
            ids = _extract_ids(result)
            self._last_write_at = time.time()
            self._last_error = None
            return MemoryWriteResult(ok=True, ids=ids)
        except Exception as exc:
            self._last_error = str(exc)
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)

    def get_all(self, scope: MemoryScope, *, limit: int | None = None) -> list[MemoryHit]:
        client = self._get_client()
        if client is None:
            return []
        try:
            result = _get_all_mem0(
                client,
                filters=build_mem0_filters(scope),
                limit=20 if limit is None else limit,
                identity=_scope_identity_kwargs(scope),
            )
            self._last_error = None
            return _normalize_mem0_results(result)
        except Exception as exc:
            self._last_error = str(exc)
            return []

    def update(self, memory_id: str, text: str) -> MemoryWriteResult:
        client = self._get_client()
        if client is None:
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)
        try:
            client.update(memory_id, data=text)
            self._last_write_at = time.time()
            self._last_error = None
            return MemoryWriteResult(ok=True, ids=[memory_id])
        except Exception as exc:
            self._last_error = str(exc)
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)

    def delete(self, memory_id: str) -> MemoryWriteResult:
        client = self._get_client()
        if client is None:
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)
        try:
            client.delete(memory_id)
            still_exists = False
            for _ in range(5):
                probe = client.get(memory_id)
                if probe is None:
                    still_exists = False
                    break
                still_exists = True
                time.sleep(0.2)
            if still_exists:
                self._last_error = (
                    f"memory {memory_id} still exists after delete confirmation"
                )
                return MemoryWriteResult(ok=False, ids=[], error=self._last_error)
            self._last_write_at = time.time()
            self._last_error = None
            return MemoryWriteResult(ok=True, ids=[memory_id])
        except Exception as exc:
            self._last_error = str(exc)
            return MemoryWriteResult(ok=False, ids=[], error=self._last_error)

    def status(self) -> MemoryBackendStatus:
        healthy = self._get_client() is not None
        try:
            from nova.agent.memory_jobs import MEMORY_JOBS

            pending_writes = MEMORY_JOBS.pending_count()
        except Exception:
            pending_writes = 0
        return MemoryBackendStatus(
            enabled=True,
            backend=self.backend_name,
            healthy=healthy,
            pending_writes=pending_writes,
            last_error=self._last_error,
            last_write_at=self._last_write_at,
        )
