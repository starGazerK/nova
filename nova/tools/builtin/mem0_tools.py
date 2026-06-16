"""
nova/tools/builtin/mem0_tools.py - Mem0 memory management tools.

These tools expose operator-style memory browsing and deletion to the model
while keeping Nova's automatic memory pipeline unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from nova.agent.memory_backend import build_memory_scope, create_memory_backend
from nova.agent.mem0_backend import Mem0MemoryBackend
from nova.tools.base import BaseTool

_PROTECTED_FILTER_KEYS = {
    "user_id",
    "agent_id",
    "run_id",
    "app_id",
    "scope",
    "workspace_id",
    "project_name",
    "session_key",
}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _error(message: str, **extra: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return _json(payload)


def _raw_mem0_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = result.get("results") or result.get("memories") or []
    elif isinstance(result, list):
        items = result
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _normalize_memory_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    memory_id = normalized.get("memory_id") or normalized.get("id")
    if memory_id:
        normalized["memory_id"] = memory_id
    return normalized


def _memory_text(item: dict[str, Any]) -> str:
    for key in ("memory", "text", "data", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _memory_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _memory_field(item: dict[str, Any], metadata: dict[str, Any], key: str) -> Any:
    return item.get(key) if item.get(key) is not None else metadata.get(key)


def _uses_limit_instead_of_top_k(exc: TypeError) -> bool:
    message = str(exc)
    return "top_k" in message and "unexpected keyword argument" in message


def _identity_kwargs_from_filters(filters: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key in ("user_id", "agent_id", "run_id"):
        value = filters.get(key)
        if value:
            identity[key] = value
    return identity


def _memory_scope_name(item: dict[str, Any]) -> str:
    metadata = _memory_metadata(item)
    scope = _memory_field(item, metadata, "scope")
    return str(scope) if isinstance(scope, str) and scope else "unknown"


def _dedupe_memory_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        memory_id = item.get("memory_id") or item.get("id")
        key = str(memory_id) if memory_id else _memory_text(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _group_memory_items(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        scope = _memory_scope_name(item)
        grouped.setdefault(scope, []).append(item)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for scope in ("user", "project", "session", "unknown"):
        if scope in grouped:
            ordered[scope] = grouped[scope]
    for scope, scope_items in grouped.items():
        if scope not in ordered:
            ordered[scope] = scope_items
    return ordered


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


class _Mem0ToolBase(BaseTool):
    def __init__(self) -> None:
        self._session_key: str | None = None
        self._channel = "cli"
        self._chat_id = "direct"

    def set_runtime_context(
        self,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        session_key: str | None = None,
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._session_key = session_key

    @staticmethod
    def _require_mem0_backend() -> Mem0MemoryBackend | None:
        backend = create_memory_backend()
        if isinstance(backend, Mem0MemoryBackend):
            return backend
        if hasattr(backend, "_get_client") and hasattr(backend, "status"):
            return backend  # type: ignore[return-value]
        return None

    def _get_client(self) -> tuple[Mem0MemoryBackend | None, Any | None, str | None]:
        backend = self._require_mem0_backend()
        if backend is None:
            return None, None, "Mem0 memory tools require NOVA_MEMORY_BACKEND=mem0 or hybrid."
        client = backend._get_client()
        if client is None:
            return backend, None, backend.status().last_error or "Mem0 backend is unavailable."
        return backend, client, None

    def _build_scope_filters(self, scope_name: str) -> dict[str, Any]:
        if scope_name == "any":
            scope = build_memory_scope(
                scope="user",
                session_key=self._session_key,
            )
            filters = {
                "user_id": scope.user_id,
                "agent_id": "nova",
                "app_id": "nova-ai",
            }
            return filters

        scope = build_memory_scope(
            scope=scope_name,  # type: ignore[arg-type]
            session_key=self._session_key,
        )
        filters: dict[str, Any] = {
            "user_id": scope.user_id,
            "agent_id": "nova",
            "app_id": "nova-ai",
            "scope": scope.scope,
        }
        if scope.scope == "session" and scope.session_key:
            filters["run_id"] = scope.session_key
            filters["session_key"] = scope.session_key
        if scope.scope in {"session", "project"}:
            filters["workspace_id"] = scope.workspace_id
            filters["project_name"] = scope.project_name
        return filters

    def _current_scope_names(self) -> list[str]:
        scopes = ["project"]
        if self._session_key:
            scopes.append("session")
        return scopes

    def _list_items_for_scope(
        self,
        client: Any,
        *,
        scope_name: str,
        limit: int,
        extra_filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        filters = self._merge_filters(
            self._build_scope_filters(scope_name),
            extra_filters,
        )
        result = _get_all_mem0(
            client,
            filters=filters,
            limit=limit,
            identity=_identity_kwargs_from_filters(filters),
        )
        items = [_normalize_memory_item(item) for item in _raw_mem0_result(result)]
        return items, filters

    def _merge_filters(
        self,
        base_filters: dict[str, Any],
        extra_filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(extra_filters, dict) or not extra_filters:
            return dict(base_filters)
        merged = dict(base_filters)
        for key, value in extra_filters.items():
            if key in _PROTECTED_FILTER_KEYS and key in merged:
                continue
            merged[key] = value
        return merged

    def _memory_matches_scope(self, memory: dict[str, Any], scope_name: str) -> bool:
        if scope_name == "any":
            return True
        metadata = _memory_metadata(memory)
        expected = self._build_scope_filters(scope_name)
        if memory.get("user_id") != expected.get("user_id"):
            return False
        if memory.get("agent_id") != expected.get("agent_id"):
            return False
        if _memory_field(memory, metadata, "app_id") != expected.get("app_id"):
            return False
        if _memory_field(memory, metadata, "scope") != expected.get("scope"):
            return False
        if scope_name in {"project", "session"}:
            if _memory_field(memory, metadata, "workspace_id") != expected.get("workspace_id"):
                return False
            if _memory_field(memory, metadata, "project_name") != expected.get("project_name"):
                return False
        if scope_name == "session":
            if memory.get("run_id") != expected.get("run_id"):
                return False
            if _memory_field(memory, metadata, "session_key") != expected.get("session_key"):
                return False
        return True

    def _fetch_memory_for_preview(self, memory_id: str, scope_name: str) -> dict[str, Any] | None:
        _backend, client, error = self._get_client()
        if error or client is None:
            return None
        result = client.get(memory_id)
        if not isinstance(result, dict):
            return None
        if not self._memory_matches_scope(result, scope_name):
            return None
        return _normalize_memory_item(result)


class SearchMemoriesTool(_Mem0ToolBase):
    @property
    def name(self) -> str:
        return "search_memories"

    @property
    def description(self) -> str:
        return (
            "Semantically search existing Mem0 memories. "
            "Supports structured filters, scope selection, and result limits."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to find."},
                "filters": {"type": "object", "description": "Additional structured filters to apply."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "scope": {
                    "type": "string",
                    "enum": ["any", "user", "project", "session"],
                    "description": "Memory scope to search. Defaults to project.",
                },
            },
            "required": ["query"],
        }

    def is_read_only(self, params: dict[str, Any] | None = None) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        _backend, client, error = self._get_client()
        if error:
            return _error(error)
        assert client is not None

        scope_name = kwargs.get("scope") or "project"
        limit = kwargs.get("limit") or 6
        filters = self._merge_filters(
            self._build_scope_filters(scope_name),
            kwargs.get("filters"),
        )
        result = _search_mem0(
            client,
            kwargs["query"],
            limit=limit,
            filters=filters,
            identity=_identity_kwargs_from_filters(filters),
        )
        items = [_normalize_memory_item(item) for item in _raw_mem0_result(result)]
        return _json({
            "ok": True,
            "scope": scope_name,
            "limit": limit,
            "filters": filters,
            "count": len(items),
            "results": items,
        })


class GetMemoriesTool(_Mem0ToolBase):
    @property
    def name(self) -> str:
        return "get_memories"

    @property
    def description(self) -> str:
        return (
            "List Mem0 memories with structured filters and lightweight pagination."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "filters": {"type": "object", "description": "Additional structured filters to apply."},
                "page": {"type": "integer", "minimum": 1, "description": "1-indexed page number."},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Memories per page."},
                "scope": {
                    "type": "string",
                    "enum": ["any", "user", "project", "session"],
                    "description": "Memory scope to browse. Defaults to current memory (project + current session). Use any for all memories.",
                },
            },
        }

    def is_read_only(self, params: dict[str, Any] | None = None) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        _backend, client, error = self._get_client()
        if error:
            return _error(error)
        assert client is not None

        scope_name = kwargs.get("scope") or ("current" if self._session_key else "project")
        page = kwargs.get("page") or 1
        page_size = kwargs.get("page_size") or 10
        start = (page - 1) * page_size
        end = start + page_size
        fetch_limit = end + 1
        filters_used: Any
        grouped_results: dict[str, list[dict[str, Any]]] | None = None

        if scope_name == "current":
            items: list[dict[str, Any]] = []
            filter_sets: list[dict[str, Any]] = []
            for current_scope in self._current_scope_names():
                scope_items, scope_filters = self._list_items_for_scope(
                    client,
                    scope_name=current_scope,
                    limit=fetch_limit,
                    extra_filters=kwargs.get("filters"),
                )
                items.extend(scope_items)
                filter_sets.append({"scope": current_scope, "filters": scope_filters})
            items = _dedupe_memory_items(items)
            filters_used = {"mode": "current", "scopes": filter_sets}
        else:
            items, filters = self._list_items_for_scope(
                client,
                scope_name=scope_name,
                limit=fetch_limit,
                extra_filters=kwargs.get("filters"),
            )
            items = _dedupe_memory_items(items)
            filters_used = filters
            if scope_name == "any":
                grouped_results = _group_memory_items(items[start:end])

        paged_items = items[start:end]
        return _json({
            "ok": True,
            "scope": scope_name,
            "filters": filters_used,
            "page": page,
            "page_size": page_size,
            "count": len(paged_items),
            "has_more": len(items) > end,
            "results": paged_items,
            "grouped_results": grouped_results,
        })


class GetMemoryTool(_Mem0ToolBase):
    @property
    def name(self) -> str:
        return "get_memory"

    @property
    def description(self) -> str:
        return "Fetch a single Mem0 memory by its memory_id."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Exact memory_id to fetch."},
                "scope": {
                    "type": "string",
                    "enum": ["any", "user", "project", "session"],
                    "description": "Optional scope check. Defaults to any.",
                },
            },
            "required": ["memory_id"],
        }

    def is_read_only(self, params: dict[str, Any] | None = None) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        _backend, client, error = self._get_client()
        if error:
            return _error(error)
        assert client is not None

        scope_name = kwargs.get("scope") or "any"
        result = client.get(kwargs["memory_id"])
        if result is None:
            return _error("Memory not found.", memory_id=kwargs["memory_id"])
        if not self._memory_matches_scope(result, scope_name):
            return _error(
                "Memory exists but is outside the requested scope.",
                memory_id=kwargs["memory_id"],
                scope=scope_name,
            )
        normalized = _normalize_memory_item(result)
        return _json({
            "ok": True,
            "scope": scope_name,
            "result": normalized,
        })


class DeleteMemoryTool(_Mem0ToolBase):
    @property
    def name(self) -> str:
        return "delete_memory"

    @property
    def description(self) -> str:
        return (
            "Delete a single Mem0 memory by memory_id. "
            "This tool always requires explicit user confirmation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Exact memory_id to delete."},
                "scope": {
                    "type": "string",
                    "enum": ["any", "user", "project", "session"],
                    "description": "Optional scope check before deletion. Defaults to any.",
                },
            },
            "required": ["memory_id"],
        }

    def permission_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(params.get("memory_id", "")).strip()
        scope_name = str(params.get("scope", "any") or "any")
        if not memory_id:
            return {}
        memory = self._fetch_memory_for_preview(memory_id, scope_name)
        if memory is None:
            return {
                "memory_id": memory_id,
                "scope": scope_name,
                "memory_preview": "(memory not found or outside requested scope)",
            }
        metadata = _memory_metadata(memory)
        actual_scope = _memory_field(memory, metadata, "scope") or scope_name
        preview = _memory_text(memory)
        if len(preview) > 500:
            preview = preview[:497].rstrip() + "..."
        return {
            "memory_id": memory_id,
            "scope": scope_name,
            "actual_scope": str(actual_scope),
            "memory_preview": preview or "(empty memory text)",
        }

    def execute(self, **kwargs: Any) -> Any:
        backend, client, error = self._get_client()
        if error:
            return _error(error)
        assert client is not None
        assert backend is not None

        scope_name = kwargs.get("scope") or "any"
        existing = client.get(kwargs["memory_id"])
        if existing is None:
            return _error("Memory not found.", memory_id=kwargs["memory_id"])
        if not self._memory_matches_scope(existing, scope_name):
            return _error(
                "Memory exists but is outside the requested scope.",
                memory_id=kwargs["memory_id"],
                scope=scope_name,
            )
        normalized = _normalize_memory_item(existing)
        result = backend.delete(kwargs["memory_id"])
        if not result.ok:
            return _error(
                result.error or "Memory deletion failed.",
                memory_id=kwargs["memory_id"],
                scope=scope_name,
            )
        try:
            from nova.agent.memory_telemetry import MEMORY_TELEMETRY

            MEMORY_TELEMETRY.record_delete(
                memory_id=kwargs["memory_id"],
                scope=scope_name,
            )
        except Exception:
            pass
        try:
            from nova.agent.memory_snapshot import refresh_memory_snapshot_silently
            from nova.agent.memory_telemetry import MEMORY_TELEMETRY

            snapshot = refresh_memory_snapshot_silently(
                backend=backend,
                session_key=self._session_key,
            )
            MEMORY_TELEMETRY.record_snapshot_refresh(
                ok=snapshot.ok,
                error=snapshot.error,
            )
        except Exception:
            pass
        return _json({
            "ok": True,
            "scope": scope_name,
            "deleted_memory": normalized,
            "memory_id": kwargs["memory_id"],
            "result": {
                "ok": result.ok,
                "ids": list(getattr(result, "ids", []) or []),
                "error": result.error,
            },
        })
