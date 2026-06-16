"""
nova/agent/memory_backend.py - Memory backend boundary for v1.1.

Phase 1 keeps Markdown as the active implementation while giving later Mem0
work a stable interface to plug into.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from nova.config import NOVA_MEMORY_BACKEND, NOVA_USER_ID, WORKDIR

MemoryScopeName = Literal["user", "session", "project"]


@dataclass(frozen=True)
class MemoryScope:
    user_id: str
    session_key: str | None
    workspace_id: str
    project_name: str
    scope: MemoryScopeName


@dataclass(frozen=True)
class MemoryHit:
    id: str | None
    text: str
    score: float | None
    metadata: dict[str, Any]
    source: str


@dataclass(frozen=True)
class MemoryWriteResult:
    ok: bool
    ids: list[str]
    error: str | None = None


@dataclass(frozen=True)
class MemoryBackendStatus:
    enabled: bool
    backend: str
    healthy: bool
    pending_writes: int
    last_error: str | None
    last_write_at: float | None


class MemoryBackend(Protocol):
    def search(self, query: str, scope: MemoryScope, *, limit: int) -> list[MemoryHit]:
        ...

    def add_messages(
        self,
        messages: list[dict],
        scope: MemoryScope,
        metadata: dict,
    ) -> MemoryWriteResult:
        ...

    def get_all(self, scope: MemoryScope, *, limit: int | None = None) -> list[MemoryHit]:
        ...

    def update(self, memory_id: str, text: str) -> MemoryWriteResult:
        ...

    def delete(self, memory_id: str) -> MemoryWriteResult:
        ...

    def status(self) -> MemoryBackendStatus:
        ...


def workspace_id_for(workspace: Path) -> str:
    """Return a stable, non-reversible ID for a workspace path."""
    resolved = str(workspace.resolve()).lower()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def build_memory_scope(
    *,
    session_key: str | None = None,
    scope: MemoryScopeName = "project",
    user_id: str = NOVA_USER_ID,
    workspace: Path = WORKDIR,
) -> MemoryScope:
    """Build the standard Nova memory scope object."""
    return MemoryScope(
        user_id=user_id,
        session_key=session_key,
        workspace_id=workspace_id_for(workspace),
        project_name=workspace.name,
        scope=scope,
    )


class MarkdownMemoryBackend:
    """Markdown-backed memory adapter preserving the existing v1.0 behavior."""

    def __init__(self, store: Any, *, backend_name: str = "markdown"):
        self.store = store
        self.backend_name = backend_name
        self._last_error: str | None = None
        self._last_write_at: float | None = None

    def _read_content(self) -> str:
        content = self.store.read_memory().strip()
        if not content or content == "(empty)":
            return ""
        return content

    def get_memory_context(self) -> str:
        content = self._read_content()
        return f"## Long-term Memory\n\n{content}" if content else ""

    def search(self, query: str, scope: MemoryScope, *, limit: int) -> list[MemoryHit]:
        content = self._read_content()
        if not content or limit <= 0:
            return []
        return [
            MemoryHit(
                id="markdown:MEMORY.md",
                text=content,
                score=None,
                metadata={
                    "scope": scope.scope,
                    "workspace_id": scope.workspace_id,
                    "project_name": scope.project_name,
                },
                source="markdown",
            )
        ]

    def add_messages(
        self,
        messages: list[dict],
        scope: MemoryScope,
        metadata: dict,
    ) -> MemoryWriteResult:
        lines = []
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content.strip()}")
        if not lines:
            return MemoryWriteResult(ok=False, ids=[], error="no message content")
        cursor = self.store.append_history("\n".join(lines))
        self._last_write_at = time.time()
        return MemoryWriteResult(ok=True, ids=[str(cursor)])

    def get_all(self, scope: MemoryScope, *, limit: int | None = None) -> list[MemoryHit]:
        return self.search("", scope, limit=1 if limit is None else limit)

    def update(self, memory_id: str, text: str) -> MemoryWriteResult:
        return MemoryWriteResult(
            ok=False,
            ids=[],
            error="markdown backend does not support exact-id updates",
        )

    def delete(self, memory_id: str) -> MemoryWriteResult:
        return MemoryWriteResult(
            ok=False,
            ids=[],
            error="markdown backend does not support exact-id deletes",
        )

    def status(self) -> MemoryBackendStatus:
        return MemoryBackendStatus(
            enabled=True,
            backend=self.backend_name,
            healthy=True,
            pending_writes=0,
            last_error=self._last_error,
            last_write_at=self._last_write_at,
        )


_MEMORY_BACKEND: MemoryBackend | None = None


def create_memory_backend(store: Any | None = None) -> MemoryBackend:
    """Create and return the process-wide memory backend singleton."""
    global _MEMORY_BACKEND
    use_singleton = store is None
    if store is None:
        if _MEMORY_BACKEND is not None:
            return _MEMORY_BACKEND
        from nova.agent.memory import _STORE

        store = _STORE

    backend_name = NOVA_MEMORY_BACKEND if NOVA_MEMORY_BACKEND else "markdown"
    if backend_name in {"mem0", "hybrid"}:
        from nova.agent.mem0_backend import Mem0MemoryBackend

        backend = Mem0MemoryBackend(backend_name=backend_name)
    else:
        backend = MarkdownMemoryBackend(store, backend_name=backend_name)
    if use_singleton:
        _MEMORY_BACKEND = backend
    return backend


def reset_memory_backend_for_tests() -> None:
    """Clear the singleton for tests that monkeypatch config or stores."""
    global _MEMORY_BACKEND
    _MEMORY_BACKEND = None
