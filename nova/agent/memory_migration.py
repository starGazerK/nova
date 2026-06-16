"""
nova/agent/memory_migration.py - One-shot legacy Markdown to Mem0 migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nova.agent.memory_backend import (
    MemoryBackend,
    MemoryWriteResult,
    build_memory_scope,
    create_memory_backend,
)
from nova.agent.memory_snapshot import export_memory_snapshot


@dataclass(frozen=True)
class MemoryMigrationResult:
    ok: bool
    migrated: int
    skipped: bool = False
    marker_path: str | None = None
    error: str | None = None


def _migration_marker(store) -> Path:
    return store.memory_dir / ".mem0_migration"


def _clean_legacy_content(content: str) -> str:
    stripped = content.strip()
    if not stripped or stripped == "(empty)":
        return ""
    return stripped


def _legacy_batches(store) -> list[tuple[str, str, str]]:
    batches: list[tuple[str, str, str]] = []
    user = _clean_legacy_content(store.read_user())
    if user:
        batches.append(("user", "legacy_USER.md", user))
    soul = _clean_legacy_content(store.read_soul())
    if soul:
        batches.append(("user", "legacy_SOUL.md", soul))
    memory = _clean_legacy_content(store.read_memory())
    if memory:
        batches.append(("project", "legacy_MEMORY.md", memory))
    return batches


def run_legacy_memory_migration(
    *,
    backend: MemoryBackend | None = None,
    store=None,
    session_key: str | None = None,
    force: bool = False,
) -> MemoryMigrationResult:
    """Migrate existing Markdown memory files into Mem0 once."""
    from nova.agent.memory import _STORE

    target_store = store or _STORE
    marker = _migration_marker(target_store)
    if marker.exists() and not force:
        return MemoryMigrationResult(
            ok=True,
            migrated=0,
            skipped=True,
            marker_path=str(marker),
        )

    target_backend = backend or create_memory_backend()
    status = target_backend.status()
    if status.backend not in {"mem0", "hybrid"}:
        return MemoryMigrationResult(
            ok=False,
            migrated=0,
            marker_path=str(marker),
            error="Mem0 migration requires NOVA_MEMORY_BACKEND=mem0 or hybrid.",
        )
    if not status.healthy:
        return MemoryMigrationResult(
            ok=False,
            migrated=0,
            marker_path=str(marker),
            error=status.last_error or "Mem0 backend is not healthy.",
        )

    migrated = 0
    errors: list[str] = []
    for scope_name, source, content in _legacy_batches(target_store):
        scope = build_memory_scope(
            scope=scope_name,  # type: ignore[arg-type]
            session_key=session_key if scope_name == "session" else None,
            workspace=target_store.workspace,
        )
        result: MemoryWriteResult = target_backend.add_messages(
            [{"role": "user", "content": content}],
            scope,
            {
                "source": "legacy_markdown_migration",
                "legacy_file": source,
            },
        )
        if result.ok:
            migrated += 1
        else:
            errors.append(f"{source}: {result.error or 'write failed'}")

    if errors:
        return MemoryMigrationResult(
            ok=False,
            migrated=migrated,
            marker_path=str(marker),
            error="; ".join(errors),
        )

    marker.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "migrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "migrated_batches": migrated,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    export_memory_snapshot(
        backend=target_backend,
        store=target_store,
        session_key=session_key,
    )
    return MemoryMigrationResult(
        ok=True,
        migrated=migrated,
        marker_path=str(marker),
    )
