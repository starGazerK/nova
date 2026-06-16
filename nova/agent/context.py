"""
nova/agent/context.py - System prompt assembly and workspace template seeding.
"""

from __future__ import annotations

import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from nova.config import (
    AGENTS_MD_PATH,
    NOVA_MEMORY_BACKEND,
    HEARTBEAT_MD_PATH,
    LEGACY_SKILLS_DIR,
    MCP_CONFIG_PATH,
    RUNTIME_DIR,
    SKILLS_DIR,
    SOUL_MD_PATH,
    TOOLS_MD_PATH,
    USER_MD_PATH,
    WORKDIR,
)

BOOTSTRAP_FILES = [ "SOUL.md","AGENTS.md", "USER.md", "TOOLS.md"]
_SEEDED_ONLY_FILES = ["HEARTBEAT.md"]
_BOOTSTRAP_PATHS = {
    "AGENTS.md": AGENTS_MD_PATH,
    "SOUL.md": SOUL_MD_PATH,
    "USER.md": USER_MD_PATH,
    "TOOLS.md": TOOLS_MD_PATH,
}
_SEEDED_ONLY_PATHS = {
    "HEARTBEAT.md": HEARTBEAT_MD_PATH,
}
_RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"
_RUNTIME_CONTEXT_END = "[/Runtime Context]"

# Location of shipped templates inside the nova package
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def seed_workspace_templates() -> None:
    """
    Copy default template files to the workspace if they don't already exist.
    Called once at startup — never overwrites user-edited files.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    for filename in BOOTSTRAP_FILES + _SEEDED_ONLY_FILES:
        src = _TEMPLATES_DIR / filename
        dst = _BOOTSTRAP_PATHS.get(filename) or _SEEDED_ONLY_PATHS[filename]
        legacy = WORKDIR / filename
        if not dst.exists() and src.exists():
            if legacy.exists():
                shutil.copy2(legacy, dst)
                print(f"[setup] Imported {filename} into .nova/")
            else:
                shutil.copy2(src, dst)
                print(f"[setup] Created .nova/{filename}")
            
    # Seed skills
    skills_src_dir = _TEMPLATES_DIR / "skills"
    skills_dst_dir = SKILLS_DIR
    if not skills_dst_dir.exists() and skills_src_dir.exists():
        shutil.copytree(skills_src_dir, skills_dst_dir)
        print("[setup] Created .nova/skills directory")
        
    # Seed MCP config
    mcp_src = _TEMPLATES_DIR / "mcp_servers.json"
    mcp_dst = MCP_CONFIG_PATH
    legacy_mcp = WORKDIR / "mcp_servers.json"
    if not mcp_dst.exists() and mcp_src.exists():
        if legacy_mcp.exists():
            shutil.copy2(legacy_mcp, mcp_dst)
            print("[setup] Imported mcp_servers.json into .nova/")
        else:
            shutil.copy2(mcp_src, mcp_dst)
            print("[setup] Created .nova/mcp_servers.json")


def build_system_prompt(skills_descriptions: str | None = None) -> str:
    """
    Assemble a rich system prompt from identity, workspace files, and skills.

    Loading order (separated by ---):
      1. Identity + Runtime info
      2. Bootstrap files (AGENTS.md, SOUL.md, USER.md, TOOLS.md from workspace)
      3. Long-term memory
      4. Active always-skills
      5. Skills summary
      6. Recent archived history
    """
    parts = []

    # 1. Identity + Runtime
    runtime = (
        f"OS: {platform.system()} {platform.release()}, "
        f"Python: {platform.python_version()}"
    )
    parts.append(
        f"# Nova\n\n"
        f"You are Nova, a coding agent.\n\n"
        f"## Runtime\n{runtime}\n\n"
        f"## Workspace\n{WORKDIR}"
    )

    # 2. Bootstrap files from Nova runtime directory
    for filename in BOOTSTRAP_FILES:
        path = _BOOTSTRAP_PATHS[filename]
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
            except Exception:
                pass

    # 3. Long-term memory
    from nova.agent.memory import _STORE as _memory

    memory_context = _get_long_term_memory_context()
    if memory_context:
        parts.append(memory_context)

    # 4. Active always-skills
    from nova.tools.registry import SKILLS as _skills

    _skills.reload()
    always_skills = _skills.get_always_skills()
    if always_skills:
        always_content = _skills.load_skills_for_context(always_skills)
        if always_content:
            parts.append(f"## Active Skills\n\n{always_content}")

    # 5. Skills summary
    summary = skills_descriptions
    if summary is None:
        summary = _skills.build_skills_summary(exclude=set(always_skills))
    if summary and summary != "(no skills)":
        parts.append(f"## Available Skills\n\n{summary}")

    # 5.5. Dynamic memory-tool guidance for Mem0-backed workspaces.
    memory_tool_guidance = _build_memory_tool_guidance()
    if memory_tool_guidance:
        parts.append(memory_tool_guidance)

    # 6. Recent archived history that has not yet been folded into MEMORY.md
    recent_history_entries = _memory.read_unprocessed_history(_memory.get_last_dream_cursor())
    if recent_history_entries:
        recent_lines = [
            f"- [{entry['timestamp']}] {entry['content']}"
            for entry in recent_history_entries[-12:]
            if entry.get("content")
        ]
        if recent_lines:
            parts.append("## Recent History\n\n" + "\n".join(recent_lines))

    return "\n\n---\n\n".join(parts)


def _build_memory_tool_guidance() -> str:
    """Teach the model when to prefer Mem0 management tools over Markdown files."""
    if NOVA_MEMORY_BACKEND not in {"mem0", "hybrid"}:
        return ""
    return (
        "## Memory Tools\n\n"
        "This workspace uses a Mem0-backed memory system.\n\n"
        "When the user explicitly asks to inspect, list, search, fetch, or delete memories, "
        "you must use the dedicated Mem0 memory tools before answering. "
        "Do not answer those requests from prior context or implicit memory recall alone.\n\n"
        "- Use `get_memories` to list current memories. If the user asks for all memories, use `scope=any`.\n"
        "- Use `search_memories` for semantic memory search by topic or fact.\n"
        "- Use `get_memory` when the user refers to a specific `memory_id`.\n"
        "- Use `delete_memory` only when the user explicitly asks to delete a specific memory; this tool requires confirmation and the confirmation prompt includes the target memory content.\n"
        "- If the user asks to delete item N from a memory list you just showed, use that listed item's `memory_id`; do not guess IDs from memory text alone.\n"
        "- Only if a Mem0 tool clearly times out or returns an error may you inspect `.nova/memory/MEMORY.md` as a fallback snapshot, and you must label it as fallback data.\n"
    )


def _get_long_term_memory_context() -> str:
    """Read long-term memory through the configured backend."""
    from nova.agent.memory_backend import create_memory_backend

    backend = create_memory_backend()
    getter = getattr(backend, "get_memory_context", None)
    if callable(getter):
        return getter()

    # Generic fallback for future backends that only implement the protocol.
    from nova.config import NOVA_MEMORY_SEARCH_LIMIT
    from nova.agent.memory_backend import build_memory_scope

    scope = build_memory_scope(scope="project")
    hits = backend.get_all(scope, limit=NOVA_MEMORY_SEARCH_LIMIT)
    lines = [hit.text.strip() for hit in hits if hit.text.strip()]
    if not lines:
        return ""
    return "## Long-term Memory\n\n" + "\n\n".join(lines)


def build_runtime_context(
    *,
    channel: str = "cli",
    chat_id: str = "direct",
    session_key: str | None = None,
    session_summary: str | None = None,
) -> str:
    """Build an untrusted runtime metadata block for the current turn."""
    lines = [
        f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Channel: {channel}",
        f"Chat ID: {chat_id}",
    ]
    if session_key:
        lines.append(f"Session Key: {session_key}")
    if session_summary:
        lines += ["", "[Resumed Session]", session_summary]
    return _RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + _RUNTIME_CONTEXT_END


def merge_runtime_context_into_messages(
    messages: list[dict[str, Any]],
    *,
    channel: str = "cli",
    chat_id: str = "direct",
    session_key: str | None = None,
    session_summary: str | None = None,
    retrieved_memory_context: str | None = None,
) -> list[dict[str, Any]]:
    """
    Inject runtime metadata into the latest user message only.

    Stored session history remains clean; only the LLM request gets the block.
    """
    if not messages:
        return []

    runtime_block = build_runtime_context(
        channel=channel,
        chat_id=chat_id,
        session_key=session_key,
        session_summary=session_summary,
    )
    merged = list(messages)

    for idx in range(len(merged) - 1, -1, -1):
        message = merged[idx]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        updated = dict(message)
        prefix_blocks = [runtime_block]
        if retrieved_memory_context:
            prefix_blocks.append(retrieved_memory_context)
        prefix = "\n\n".join(prefix_blocks)
        if isinstance(content, str):
            updated["content"] = f"{prefix}\n\n{content}"
        elif isinstance(content, list):
            injected = [{"type": "text", "text": block} for block in prefix_blocks]
            updated["content"] = injected + content
        else:
            updated["content"] = f"{prefix}\n\n{content}"
        merged[idx] = updated
        break
    return merged
