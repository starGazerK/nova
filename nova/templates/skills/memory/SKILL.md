---
name: memory
description: Use Nova's two-layer memory system and search archived history safely.
always: true
---

# Memory

## Structure

- `.nova/SOUL.md` - bot personality and communication style. Managed by Dream.
- `.nova/USER.md` - user profile and durable preferences. Managed by Dream.
- `.nova/memory/history.jsonl` - append-only JSONL archive. It is not fully loaded into context.
- The optional readable memory snapshot export is for operators only, not model-side memory lookup.

## Search Past Events

Use `grep`/`rg` over `.nova/memory/history.jsonl` when you need old details not present in the current context.

- Start broad with a count or files-with-matches search before expanding output.
- Use content mode with nearby context when you need exact lines.
- Use fixed-string matching for timestamps, cursor ids, paths, or JSON fragments.
- Prefer searching `.nova/memory/*.jsonl` instead of loading the whole file.

Examples:

```text
grep(pattern="keyword", path=".nova/memory/history.jsonl", case_insensitive=true)
grep(pattern="2026-04-02 10:00", path=".nova/memory/history.jsonl", fixed_strings=true)
grep(pattern="oauth|token", path=".nova/memory", glob="*.jsonl", output_mode="content", case_insensitive=true)
```

## Important

- In Mem0-backed workspaces, use the dedicated Mem0 memory tools for explicit memory listing/searching/fetching/deleting requests before answering; do not rely on implicit recall alone.
- Only if a Mem0 tool clearly times out or errors may you inspect `.nova/memory/MEMORY.md` as a fallback snapshot, and you must say it is fallback data.
- Do not edit `.nova/SOUL.md`, `.nova/USER.md`, or snapshot export files directly unless the user explicitly asks.
- Use `/memory` to manually run Dream consolidation.
- Use `/memory export` only when the user explicitly wants a readable snapshot export.
- Use `/dream-log` to inspect Dream changes and `/dream-restore` to roll one back.
