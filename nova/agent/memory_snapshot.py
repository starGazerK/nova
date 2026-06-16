"""
nova/agent/memory_snapshot.py - Export Mem0 memories to a readable snapshot.

The snapshot is a readable operator cache and controlled fallback. Mem0 remains
the source of truth when the Mem0 backend is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from nova.agent.memory_backend import (
    MemoryBackend,
    MemoryHit,
    build_memory_scope,
    create_memory_backend,
)
from nova.config import MEM0_SNAPSHOT_LIMIT, WORKDIR


@dataclass(frozen=True)
class MemorySnapshotResult:
    ok: bool
    path: str
    count: int
    error: str | None = None


def _clean_snapshot_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned


def _dedupe_hits(hits: Iterable[MemoryHit], *, limit: int) -> list[MemoryHit]:
    if limit <= 0:
        return []
    seen: set[str] = set()
    kept: list[MemoryHit] = []
    for hit in hits:
        text = _clean_snapshot_text(hit.text)
        if not text:
            continue
        key = hit.id or text.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept


def _group_hits_by_scope(hits: Iterable[MemoryHit]) -> tuple[list[MemoryHit], list[MemoryHit], list[MemoryHit]]:
    grouped = {"user": [], "project": [], "session": []}
    for hit in hits:
        scope = str(hit.metadata.get("scope") or "").strip().lower()
        if scope in grouped:
            grouped[scope].append(hit)
    return grouped["user"], grouped["project"], grouped["session"]


def _ensure_scope_metadata(hits: Iterable[MemoryHit], scope_name: str) -> list[MemoryHit]:
    normalized: list[MemoryHit] = []
    for hit in hits:
        metadata = dict(hit.metadata or {})
        metadata.setdefault("scope", scope_name)
        normalized.append(MemoryHit(
            id=hit.id,
            text=hit.text,
            score=hit.score,
            metadata=metadata,
            source=hit.source,
        ))
    return normalized


def render_memory_snapshot(
    *,
    user_hits: list[MemoryHit],
    project_hits: list[MemoryHit],
    session_hits: list[MemoryHit],
    generated_at: datetime | None = None,
    limit: int = MEM0_SNAPSHOT_LIMIT,
) -> str:
    """Render a deterministic Markdown snapshot from normalized backend hits."""
    ts = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")

    remaining = max(0, limit)
    user = _dedupe_hits(user_hits, limit=remaining)
    remaining -= len(user)
    project = _dedupe_hits(project_hits, limit=remaining)
    remaining -= len(project)
    session = _dedupe_hits(session_hits, limit=remaining)

    def _section(title: str, hits: list[MemoryHit]) -> list[str]:
        lines = [f"## {title}", ""]
        if hits:
            lines.extend(f"- {_clean_snapshot_text(hit.text)}" for hit in hits)
        else:
            lines.append("- (empty)")
        return lines

    lines = [
        "# Memory Snapshot",
        "",
        f"Generated from Mem0 at {ts}.",
        "This file is a readable cache. Mem0 is the source of truth when enabled.",
        "",
        *_section("User Memory", user),
        "",
        *_section("Project Memory", project),
        "",
        *_section("Recent Session Memory", session),
        "",
    ]
    return "\n".join(lines)


def parse_memory_snapshot(
    content: str,
    *,
    limit: int = MEM0_SNAPSHOT_LIMIT,
) -> list[MemoryHit]:
    """Parse a snapshot file back into lightweight hits for fallback reads."""
    if limit <= 0:
        return []

    scope_by_section = {
        "User Memory": "user",
        "Project Memory": "project",
        "Recent Session Memory": "session",
    }
    hits: list[MemoryHit] = []
    current_scope = "snapshot"

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_scope = scope_by_section.get(line[3:].strip(), "snapshot")
            continue
        if not line.startswith("- "):
            continue
        text = _clean_snapshot_text(line[2:])
        if not text or text == "(empty)":
            continue
        hits.append(MemoryHit(
            id=f"snapshot:{current_scope}:{len(hits) + 1}",
            text=text,
            score=None,
            metadata={
                "scope": current_scope,
                "source": "snapshot_fallback",
            },
            source="snapshot",
        ))
        if len(hits) >= limit:
            break
    return hits


def load_snapshot_hits(
    *,
    store=None,
    limit: int = MEM0_SNAPSHOT_LIMIT,
) -> list[MemoryHit]:
    """Load parsed hits from the current MEMORY.md snapshot file."""
    from nova.agent.memory import _STORE

    target_store = store or _STORE
    try:
        content = target_store.read_memory().strip()
    except Exception:
        return []
    if not content or content == "(empty)":
        return []
    return parse_memory_snapshot(content, limit=limit)


def _workspace_snapshot_hits(
    *,
    backend: MemoryBackend,
    workspace,
    limit: int,
) -> tuple[list[MemoryHit], str | None]:
    """Collect snapshot candidates for the whole workspace across scopes."""
    from nova.agent.mem0_backend import Mem0MemoryBackend
    from nova.agent.mem0_backend import _normalize_mem0_results

    if isinstance(backend, Mem0MemoryBackend):
        client = backend._get_client()
        if client is None:
            return [], backend.status().last_error
        user_scope = build_memory_scope(scope="user", workspace=workspace)
        filters = {
            "app_id": "nova-ai",
            "workspace_id": user_scope.workspace_id,
            "project_name": user_scope.project_name,
        }
        identity = {
            "user_id": user_scope.user_id,
            "agent_id": "nova",
        }
        try:
            try:
                result = client.get_all(filters=filters, top_k=limit, **identity)
            except TypeError:
                result = client.get_all(filters=filters, limit=limit, **identity)
            hits = _normalize_mem0_results(result)
        except Exception as exc:
            return [], str(exc)
        if not hits:
            return [], None
        return hits, None

    user_scope = build_memory_scope(scope="user", workspace=workspace)
    project_scope = build_memory_scope(scope="project", workspace=workspace)
    session_scope = build_memory_scope(
        scope="session",
        session_key="snapshot",
        workspace=workspace,
    )
    user_hits = _ensure_scope_metadata(backend.get_all(user_scope, limit=limit), "user")
    project_hits = _ensure_scope_metadata(backend.get_all(project_scope, limit=limit), "project")
    session_hits = _ensure_scope_metadata(backend.get_all(session_scope, limit=limit), "session")
    return [*user_hits, *project_hits, *session_hits], None


def export_memory_snapshot(
    *,
    backend: MemoryBackend | None = None,
    store=None,
    session_key: str | None = None,
    limit: int = MEM0_SNAPSHOT_LIMIT,
    commit: bool = True,
) -> MemorySnapshotResult:
    """Refresh `.nova/memory/MEMORY.md` from the configured Mem0 backend."""
    from nova.agent.memory import _STORE

    target_store = store or _STORE
    target_backend = backend or create_memory_backend()
    try:
        path = target_store.memory_file
        all_hits, error = _workspace_snapshot_hits(
            backend=target_backend,
            workspace=target_store.workspace,
            limit=limit,
        )
        if error:
            raise RuntimeError(error)
        user_hits, project_hits, session_hits = _group_hits_by_scope(all_hits)
        content = render_memory_snapshot(
            user_hits=user_hits,
            project_hits=project_hits,
            session_hits=session_hits,
            limit=limit,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if commit:
            target_store.ensure_git_initialized()
            target_store.git.auto_commit("memory: export mem0 snapshot")
        return MemorySnapshotResult(
            ok=True,
            path=str(path),
            count=len(_dedupe_hits(all_hits, limit=limit)),
        )
    except Exception as exc:
        fallback = getattr(target_store, "memory_file", "")
        return MemorySnapshotResult(ok=False, path=str(fallback), count=0, error=str(exc))


def refresh_memory_snapshot_silently(
    *,
    backend: MemoryBackend | None = None,
    store=None,
    session_key: str | None = None,
    limit: int = MEM0_SNAPSHOT_LIMIT,
) -> MemorySnapshotResult:
    """Best-effort snapshot refresh after Mem0 state changes."""
    return export_memory_snapshot(
        backend=backend,
        store=store,
        session_key=session_key,
        limit=limit,
        commit=False,
    )
