# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## bash — Safety

- Commands timeout at 120s
- Dangerous commands (rm -rf /, sudo, shutdown, reboot) are blocked
- URLs targeting internal networks are blocked (SSRF protection)
- Output is truncated at 50,000 characters

## read_file / write_file / edit_file

- Paths are sandboxed to the workspace directory
- Always read a file before editing it
- edit_file replaces the first occurrence of old_text only

## task (subagent)

- "Explore" type: isolated read-only subagent task
- "general-purpose" type: isolated builder subagent task
- Subagents have their own conversation state, transcript, and output file
- Use `check_subagent`, `list_subagents`, `wait_subagent`, `control_subagent`, or `task_output` to inspect/control them after launch

## background_run / check_background / task_output

- Runs commands in a background thread (default timeout 120s)
- Use check_background with the returned task_id to poll status
- Use task_output to wait for completion or read the task's output/log path
- Background task notifications are automatically drained each turn

## Permissions

- Sensitive tools (bash, file writes/edits, background_run) require approval
- Workspace-internal write/edit are auto-allowed by default
- Bash is gated by program-name allowlist plus per-prefix rules
- Use `/permissions` in REPL to inspect rules; persisted at `.nova/permissions.json`

## cron

- Schedules one-shot or recurring agent tasks
- `add` supports `every_seconds`, `at`, or `cron_expr`
- Cron expressions require optional dependency `croniter`
- Jobs are persisted in `.nova/cron/jobs.json`

## task_create / task_update / task_list

- File-backed persistent tasks in .tasks/ directory
- Support dependencies (blockedBy, blocks)
- Use for multi-step work tracking

## TodoWrite

- In-memory checklist – lighter than file tasks
- Only one item can be in_progress at a time
- Max 20 items

## Mem0 memory tools

- When `NOVA_MEMORY_BACKEND=mem0` or `hybrid`, and the user explicitly asks to list/search/inspect/delete memories, you must call the Mem0 tool first and should not answer from implicit recall alone
- Use `get_memories` without `scope` to list current memory (`project` + current `session`); use `scope=any` when the user asks for all memories
- Use `search_memories` for semantic search with optional filters and result limits
- Use `get_memory` to fetch one memory by `memory_id`
- Use `delete_memory` only for explicit deletion requests; it always requires confirmation and the confirmation prompt includes the target memory content
- If the user asks to delete item N from a memory list you just showed, use that listed item's `memory_id`; do not guess IDs from memory text alone
- Only if a Mem0 tool clearly times out or returns an error may you inspect `.nova/memory/MEMORY.md` as a fallback snapshot, and you must treat it as potentially stale
