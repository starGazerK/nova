"""
nova/cli/repl.py - Interactive REPL (async) with Rich UI and prompt_toolkit.

REPL commands: /new /sessions /resume /compact /memory /dream-log /dream-restore /cron /heartbeat /mcp /tasks /bg /subagents /permissions /status /help
"""

import asyncio
import html
import json
import signal
import shlex
import textwrap
import time
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from nova.agent.compression import auto_compact, estimate_tokens, extract_session_summary
from nova.agent.context import build_system_prompt, seed_workspace_templates
from nova.agent.loop import agent_loop, get_autocompact
from nova.agent.memory import MemoryStore, cleanup_memory_files_once, consolidate_memory
from nova.agent.memory_migration import run_legacy_memory_migration
from nova.agent.memory_snapshot import export_memory_snapshot
from nova.agent.memory_status import get_memory_operator_status, render_memory_status_lines
from nova.agent.memory_backend import create_memory_backend
from nova.cron.service import _get_croniter
from nova.cron.types import CronSchedule
from nova.config import (
    HEARTBEAT_INTERVAL_SECONDS,
    IDLE_COMPACT_MINUTES,
    LEGACY_SESSION_DIR,
    MCP_CONFIG_PATH,
    MEMORY_CONSOLIDATION_INTERVAL_SECONDS,
    MODEL,
    MODEL_TEMPERATURE,
    NOVA_MEMORY_BACKEND,
    RUNTIME_DIR,
    SESSION_DIR,
    TOKEN_THRESHOLD,
    WORKDIR,
)
from nova.cron.types import CronJob, CronPayload
from nova.heartbeat.service import HeartbeatService
from nova.mcp.loader import load_mcp
from nova.session.store import SessionStore
from nova.terminal_ui import (
    ASSISTANT_LEFT_PADDING,
    BANNER_SIDE_RESERVE,
    MAX_CONTENT_WIDTH,
    content_width,
    sync_console_width,
)
from nova.tools.builtin.ask import set_ask_handler
from nova.tools.registry import (
    BG,
    CRON,
    SKILLS,
    SUBAGENT,
    TASK_MGR,
    TODO,
    TOOL_HANDLERS,
    TOOLS,
    set_permission_prompt_handler,
)

console = Console()
_MEMORY = MemoryStore(Path.cwd())

_HISTORY_PATH = Path.home() / ".nova" / "cli_history"
_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
_prompt_session: PromptSession | None = None
_PROMPT_CONTINUATION = " " * 7
_HISTORY_BOX_WIDTH = MAX_CONTENT_WIDTH


class _GenerationInterrupted(Exception):
    """Raised when Ctrl+C cancels the active model generation."""


class _InterruptController:
    """Route Ctrl+C to the active generation task instead of exiting the REPL."""

    def __init__(self) -> None:
        self.active_task: asyncio.Task | None = None
        self.interrupted = False
        self._previous_handler = None

    def install(self) -> None:
        try:
            self._previous_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        except Exception:
            self._previous_handler = None

    def restore(self) -> None:
        if self._previous_handler is None:
            return
        try:
            signal.signal(signal.SIGINT, self._previous_handler)
        except Exception:
            pass

    def _handle_sigint(self, signum, frame) -> None:
        task = self.active_task
        if task is not None and not task.done():
            self.interrupted = True
            task.cancel()
            return
        raise KeyboardInterrupt

    async def run_interruptible(self, coro):
        task = asyncio.create_task(coro)
        self.active_task = task
        self.interrupted = False
        try:
            return await task
        except asyncio.CancelledError:
            if self.interrupted:
                raise _GenerationInterrupted
            raise
        finally:
            if self.active_task is task:
                self.active_task = None
            self.interrupted = False


def _get_prompt_session() -> PromptSession:
    global _prompt_session
    if _prompt_session is None:
        sync_console_width(console)
        _prompt_session = PromptSession(
            history=FileHistory(str(_HISTORY_PATH)),
            style=Style.from_dict({
                "": "ansibrightcyan",
                "bottom-toolbar": "ansibrightblack",
            }),
        )
    return _prompt_session


def _prompt_key_bindings() -> KeyBindings:
    """Use Enter to submit and Control-J as the multiline newline fallback."""
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def _format_count(value: int | float) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _context_stats(history: list[dict] | None = None) -> tuple[int, int, float]:
    used = estimate_tokens(history or [])
    total = max(TOKEN_THRESHOLD, 1)
    ratio = min(999.0, (used / total) * 100)
    return used, total, ratio


def _prompt_stats_text(history: list[dict] | None = None) -> str:
    used, total, ratio = _context_stats(history)
    context_left = max(0.0, 100.0 - ratio)
    return (
        f"ctx {_format_count(used)}/{_format_count(total)} ({ratio:.1f}%)   "
        f"{MODEL} ({context_left:.0f}% context left)"
    )


def _prompt_rule_line() -> str:
    return "-" * content_width()


def _render_prompt_rule() -> None:
    print_formatted_text(
        HTML(f"<ansibrightcyan>{html.escape(_prompt_rule_line())}</ansibrightcyan>")
    )


def _prompt_line_html() -> HTML:
    return HTML("<ansibrightcyan>user &gt;</ansibrightcyan> ")


def _prompt_stats_toolbar(history: list[dict] | None = None) -> HTML:
    return HTML(f"<ansibrightblack>{html.escape(_prompt_stats_text(history))}</ansibrightblack>")


def _render_prompt_gap() -> None:
    """Leave two blank lines before the next input prompt."""
    print_formatted_text("")
    print_formatted_text("")


async def _ask_user(history: list[dict] | None = None) -> str:
    """Prompt the user for a single line; returns stripped text."""
    with patch_stdout():
        _render_prompt_gap()
        sync_console_width(console)
        line = await _get_prompt_session().prompt_async(
            _prompt_line_html(),
            bottom_toolbar=_prompt_stats_toolbar(history),
            key_bindings=_prompt_key_bindings(),
            multiline=True,
            prompt_continuation=HTML(
                f"<ansibrightcyan>{html.escape(_PROMPT_CONTINUATION)}</ansibrightcyan>"
            ),
            wrap_lines=True,
        )
        _render_prompt_rule()
    return (line or "").strip()


def _rollback_interrupted_turn(
    store: SessionStore,
    session_key: str,
    history: list[dict],
) -> None:
    """Drop the pending user turn after Ctrl+C cancels a generation."""
    state = store.load_state(session_key, restore_interrupted=False)
    metadata = dict(state.get("metadata") or {})
    if metadata.get("pending_user_turn"):
        metadata.pop("pending_user_turn", None)
        metadata.pop("runtime_checkpoint", None)
        state["metadata"] = metadata
        messages = list(state.get("messages") or [])
        if messages and messages[-1].get("role") == "user":
            messages.pop()
        state["messages"] = messages
        state["updated_at"] = datetime.now().astimezone()
        store.save_state(session_key, state)
        history[:] = messages


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can display it safely."""
    sync_console_width(console)
    ansi_console = Console(
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


async def _interactive_print(render_fn) -> None:
    """Print async updates without corrupting the active prompt line."""
    def _write() -> None:
        ansi = _render_interactive_ansi(render_fn)
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _interactive_notice(text: str) -> None:
    await _interactive_print(lambda c: c.print(f"[dim]{escape(text)}[/dim]"))


async def _interactive_response(label: str, body: str) -> None:
    sync_console_width(console)
    safe_label = escape(label)
    safe_body = escape(body)
    await _interactive_print(
        lambda c: (
            c.print(f"[cyan]{safe_label}[/cyan]"),
            c.print(safe_body, highlight=False, soft_wrap=True),
            c.print(),
        )
    )


async def _permission_prompt(request: dict) -> dict | None:
    message = str(request.get("message", "")).strip()
    await _interactive_print(
        lambda c: (
            c.print("[yellow]Permission required[/yellow]"),
            c.print(escape(message), highlight=False, soft_wrap=True),
            c.print("[dim]Allow once: y | session rule: s | persist rule: a | deny: n[/dim]"),
            c.print(),
        )
    )

    while True:
        with patch_stdout():
            line = await _get_prompt_session().prompt_async(
                HTML("<b><ansiyellow>Approval:</ansiyellow></b> ")
            )
        choice = (line or "").strip().lower()
        if choice in {"y", "yes"}:
            return {"action": "allow"}
        scope = str(request.get("scope_hint") or "allow_tool")
        if choice in {"s", "session"}:
            return {"action": "allow", "scope": scope, "persist": False}
        if choice in {"a", "always"}:
            return {"action": "allow", "scope": scope, "persist": True}
        if choice in {"n", "no", ""}:
            return {"action": "deny"}
        await _interactive_notice("Use y / s / a / n.")


async def _ask_user_handler(question: str, options: list[str] | None) -> str:
    """Interactive ask_user prompt with arrow-key option picker."""
    safe_q = escape(question)
    if options:
        selected_index = 0

        def _render_ask():
            fragments: list[tuple[str, str]] = [
                ("class:title", f"{question}\n\n"),
                ("class:hint", "Up/Down select, Enter confirm, Esc/type for free text.\n\n"),
            ]
            for idx, opt in enumerate(options):
                pointer = "> " if idx == selected_index else "  "
                style = "class:selected" if idx == selected_index else "class:item"
                fragments.append((style, f"{pointer}{opt}\n"))
            return fragments

        body = FormattedTextControl(_render_ask, focusable=True, show_cursor=False)
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _move_up(event) -> None:
            nonlocal selected_index
            selected_index = (selected_index - 1) % len(options)
            event.app.invalidate()

        @kb.add("down")
        @kb.add("j")
        def _move_down(event) -> None:
            nonlocal selected_index
            selected_index = (selected_index + 1) % len(options)
            event.app.invalidate()

        @kb.add("enter")
        def _accept(event) -> None:
            event.app.exit(result=options[selected_index])

        @kb.add("escape")
        @kb.add("c-c")
        def _free_text(event) -> None:
            event.app.exit(result=None)

        app = Application(
            layout=Layout(HSplit([Window(content=body, always_hide_cursor=True)])),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            style=Style.from_dict({
                "title": "bold",
                "hint": "ansibrightblack",
                "item": "",
                "selected": "reverse bold",
            }),
        )

        with patch_stdout():
            result = await app.run_async()

        if result is not None:
            return result

        # User pressed Esc — fall through to free-text input

    await _interactive_print(
        lambda c: (
            c.print(f"[bold]{safe_q}[/bold]"),
            c.print(),
        )
    )
    with patch_stdout():
        _render_prompt_gap()
        sync_console_width(console)
        line = await _get_prompt_session().prompt_async(
            _prompt_line_html(),
            bottom_toolbar=_prompt_stats_toolbar(),
            key_bindings=_prompt_key_bindings(),
            multiline=True,
            prompt_continuation=HTML(
                f"<ansibrightcyan>{html.escape(_PROMPT_CONTINUATION)}</ansibrightcyan>"
            ),
            wrap_lines=True,
        )
        _render_prompt_rule()
    return (line or "").strip() or "(no response)"
_HELP_TEXT = """[bold]Nova commands:[/bold]
  /new                     Start a new conversation
  /sessions                List saved sessions
  /resume                  Interactively pick a session to resume
  /resume <#|key>          Resume a specific session
  /compact                 Compress conversation context
  /memory                  Run memory consolidation now
  /memory status           Show memory backend and write queue status
  /memory export           Refresh Markdown snapshot from Mem0
  /memory migrate          Migrate legacy Markdown memory into Mem0 once
  /dream-log               Show the latest Dream memory change
  /dream-log <sha>         Show a specific Dream memory change
  /dream-restore           List recent Dream memory versions
  /dream-restore <sha>     Restore a Dream memory version
  /cron                    Show cron jobs and service state
  /cron run <id>           Run a job immediately
  /cron rm <id>            Remove a job
  /cron on/off <id>        Enable or disable a job
  /cron add every <seconds> <message>
  /cron add at <iso-datetime> <message>
  /cron add expr <cron-expr> <message> [tz]
  /heartbeat               Trigger one heartbeat tick now
  /mcp                     Show MCP servers and loaded capabilities
  /tasks                   Show task board
  /bg                      Show all background tasks
  /bg <id>                 Show one background task
  /bg output <id>          Show task output
  /subagents               List subagents
  /subagents <id>          Show subagent details
  /subagents output <id>
  /subagents transcript <id>
  /subagents fg <id>       Move subagent to foreground
  /subagents bg <id>       Move subagent to background
  /subagents stop <id> [reason]
  /permissions             Show permission rules (persisted + session)
  /status                  Show current session info
  /help                    Show this help
  /exit                    Quit"""


_REPLAY_TAIL = 10  # How many visible turns to replay on /resume
_MEMORY_JOB_ID = "memory_consolidation"
_SESSION_PICK_LIMIT = 20


def _extract_changed_files(diff: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


def _format_dream_log_content(commit, diff: str, *, requested_sha: str | None = None) -> str:
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {_format_changed_files(diff)}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend(["", "Dream recorded this version, but there is no file diff to display."])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for commit in commits:
        lines.append(f"- `{commit.sha}` {commit.timestamp} - {commit.message.splitlines()[0]}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


def _handle_dream_log_command(query: str) -> None:
    _MEMORY.ensure_git_initialized()
    git = _MEMORY.git
    if not git.is_initialized():
        console.print("[dim]  Dream history is not available because memory versioning is not initialized.[/dim]")
        return

    parts = shlex.split(query)
    if len(parts) > 1:
        sha = parts[1]
        result = git.show_commit_diff(sha)
        if not result:
            console.print(f"[dim]  Couldn't find Dream change {sha}.[/dim]")
            return
        commit, diff = result
        console.print(_format_dream_log_content(commit, diff, requested_sha=sha), highlight=False)
        return

    commits = git.log(max_entries=1)
    if not commits:
        console.print("[dim]  Dream memory has no saved versions yet.[/dim]")
        return
    result = git.show_commit_diff(commits[0].sha)
    if not result:
        console.print("[dim]  Dream memory has no diff to display yet.[/dim]")
        return
    commit, diff = result
    console.print(_format_dream_log_content(commit, diff), highlight=False)


def _handle_dream_restore_command(query: str) -> None:
    _MEMORY.ensure_git_initialized()
    git = _MEMORY.git
    if not git.is_initialized():
        console.print("[dim]  Dream history is not available because memory versioning is not initialized.[/dim]")
        return

    parts = shlex.split(query)
    if len(parts) == 1:
        commits = git.log(max_entries=10)
        if not commits:
            console.print("[dim]  Dream memory has no saved versions to restore yet.[/dim]")
            return
        console.print(_format_dream_restore_list(commits), highlight=False)
        return

    sha = parts[1]
    result = git.show_commit_diff(sha)
    changed_files = _format_changed_files(result[1]) if result else "the tracked memory files"
    new_sha = git.revert(sha)
    if new_sha:
        console.print(
            (
                f"Restored Dream memory to the state before `{sha}`.\n\n"
                f"- New safety commit: `{new_sha}`\n"
                f"- Restored files: {changed_files}\n\n"
                f"Use `/dream-log {new_sha}` to inspect the restore diff."
            ),
            highlight=False,
        )
    else:
        console.print(f"[dim]  Couldn't restore Dream change {sha}.[/dim]")


def _render_mcp_startup(mcp_client) -> None:
    """Print a concise MCP startup summary."""
    for line in mcp_client.startup_summary_lines():
        console.print(f"[dim]{line}[/dim]")


def _render_mcp_details(mcp_client) -> None:
    """Print detailed MCP summary including capability names."""
    for line in mcp_client.detailed_summary_lines():
        console.print(f"[dim]{line}[/dim]")


def _render_status_panel(
    *,
    session_key: str,
    history: list[dict],
    session_summary: str | None,
    heartbeat,
    mcp_client,
) -> None:
    sync_console_width(console)
    console.print("[bold cyan]Status[/bold cyan]")
    console.print(f"[dim]Session[/dim]   {session_key}")
    console.print(f"[dim]Messages[/dim]  {len(history)}")
    console.print(f"[dim]Model[/dim]     {MODEL}")
    console.print(f"[dim]Summary[/dim]   {'yes' if session_summary else 'none'}")

    cron_lines = _format_cron_snapshot()
    if cron_lines:
        console.print()
        console.print("[bold cyan]Cron[/bold cyan]")
        for line in cron_lines:
            console.print(f"[dim]{line.strip()}[/dim]")

    hb = heartbeat.status()
    console.print()
    console.print("[bold cyan]Heartbeat[/bold cyan]")
    console.print(
        f"[dim]State[/dim]     {'running' if hb['running'] else 'stopped'}"
    )
    console.print(f"[dim]Interval[/dim]  every {hb['interval_s']}s")
    console.print(f"[dim]File[/dim]      {'yes' if hb['present'] else 'no'}")
    if hb.get("last_action"):
        console.print(
            f"[dim]Last[/dim]      {hb['last_action']} ({hb.get('last_reason') or 'n/a'})"
        )

    memory_lines = render_memory_status_lines(
        get_memory_operator_status(store=_MEMORY),
        detailed=False,
    )
    if memory_lines:
        console.print()
        console.print("[bold cyan]Memory[/bold cyan]")
        for line in memory_lines:
            if ":" in line:
                label, value = line.split(":", 1)
                console.print(f"[dim]{label}:[/dim]{value}")
            else:
                console.print(f"[dim]{line}[/dim]")

    console.print()
    console.print("[bold cyan]MCP[/bold cyan]")
    if mcp_client and mcp_client.connected_servers:
        console.print(
            f"[dim]Servers[/dim]   {len(mcp_client.connected_servers)}"
        )
        console.print(
            f"[dim]Capabilities[/dim] {len(mcp_client.tool_schemas)}"
        )
        for line in mcp_client.startup_summary_lines():
            text = line.replace("[mcp] ", "", 1)
            console.print(f"[dim]{text}[/dim]")
    else:
        console.print("[dim]none[/dim]")


def _render_mcp_details_panel(mcp_client) -> None:
    sync_console_width(console)
    if not mcp_client or not mcp_client.connected_servers:
        console.print("[dim]No MCP servers connected.[/dim]")
        return

    console.print("[bold cyan]MCP Servers[/bold cyan]")
    for server_name in mcp_client.connected_servers:
        caps = mcp_client.server_capabilities.get(
            server_name,
            {"tools": [], "resources": [], "prompts": []},
        )
        tools = [mcp_client._display_name(name) for name in caps.get("tools", [])]
        resources = [mcp_client._display_name(name) for name in caps.get("resources", [])]
        prompts = [mcp_client._display_name(name) for name in caps.get("prompts", [])]
        total = len(tools) + len(resources) + len(prompts)

        console.print()
        console.print(f"[bold white]{server_name}[/bold white] [dim]({total} capabilities)[/dim]")
        console.print(
            f"[dim]tools[/dim]      {len(tools)}    "
            f"[dim]resources[/dim]  {len(resources)}    "
            f"[dim]prompts[/dim]  {len(prompts)}"
        )
        if tools:
            console.print(f"[dim]  tools:[/dim] {', '.join(tools)}")
        if resources:
            console.print(f"[dim]  resources:[/dim] {', '.join(resources)}")
        if prompts:
            console.print(f"[dim]  prompts:[/dim] {', '.join(prompts)}")


def _format_cron_snapshot() -> list[str]:
    status = CRON.status(include_system=False)
    lines = [
        f"  Cron: {'running' if status['enabled'] else 'stopped'}, {status['jobs']} job(s)"
    ]
    for job in CRON.list_jobs(include_disabled=True, include_system=False):
        state_bits = []
        if job.state.next_run_at_ms:
            state_bits.append(
                "next " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.state.next_run_at_ms / 1000))
            )
        if job.state.last_status:
            state_bits.append(f"last {job.state.last_status}")
        state = f" [{' | '.join(state_bits)}]" if state_bits else ""
        lines.append(f"  - {job.id} {job.name} ({'on' if job.enabled else 'off'}){state}")
    return lines


def _render_cron_table() -> None:
    jobs = CRON.list_jobs(include_disabled=True, include_system=False)
    status = CRON.status(include_system=False)
    console.print(
        f"[dim]  Cron: {'running' if status['enabled'] else 'stopped'}, "
        f"{status['jobs']} job(s)[/dim]"
    )
    if not jobs:
        console.print("[dim]  No scheduled jobs.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("State", no_wrap=True)
    table.add_column("Message")

    for job in jobs:
        if job.schedule.kind == "every" and job.schedule.every_ms:
            schedule = f"every {job.schedule.every_ms // 1000}s"
        elif job.schedule.kind == "cron":
            tz = f" {job.schedule.tz}" if job.schedule.tz else ""
            schedule = f"{job.schedule.expr}{tz}"
        elif job.schedule.kind == "at" and job.schedule.at_ms:
            schedule = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.schedule.at_ms / 1000))
        else:
            schedule = job.schedule.kind

        state_parts = ["on" if job.enabled else "off"]
        if job.state.next_run_at_ms:
            state_parts.append(
                "next " + time.strftime("%m-%d %H:%M:%S", time.localtime(job.state.next_run_at_ms / 1000))
            )
        if job.state.last_status:
            state_parts.append(f"last {job.state.last_status}")
        message = " ".join(job.payload.message.split())
        if len(message) > 56:
            message = message[:53] + "..."
        table.add_row(job.id, job.name, schedule, " | ".join(state_parts), message)

    console.print(table)


async def _handle_cron_command(query: str) -> None:
    parts = shlex.split(query)
    if len(parts) == 1:
        _render_cron_table()
        return

    sub = parts[1].lower()
    if sub in {"list", "ls"}:
        _render_cron_table()
        return
    if sub == "status":
        for line in _format_cron_snapshot():
            console.print(f"[dim]{line}[/dim]")
        return
    if sub in {"rm", "remove", "run", "on", "off"}:
        if len(parts) < 3:
            console.print(f"[dim]  Usage: /cron {sub} <job_id>[/dim]")
            return
        job_id = parts[2]
        if sub in {"rm", "remove"}:
            result = CRON.remove_job(job_id)
            if result == "removed":
                msg = "Removed " + job_id
            elif result == "protected":
                msg = "Job is system-managed: " + job_id
            else:
                msg = "Job not found: " + job_id
            console.print(f"[dim]  {msg}[/dim]")
            return
        if sub == "run":
            ran = await CRON.run_job(job_id, force=True)
            console.print(f"[dim]  {'Ran ' + job_id if ran else 'Job not found: ' + job_id}[/dim]")
            return
        if sub == "on":
            job = CRON.enable_job(job_id, True)
            if job == "protected":
                msg = "Job is system-managed: " + job_id
            else:
                msg = "Enabled " + job_id if job else "Job not found: " + job_id
            console.print(f"[dim]  {msg}[/dim]")
            return
        if sub == "off":
            job = CRON.enable_job(job_id, False)
            if job == "protected":
                msg = "Job is system-managed: " + job_id
            else:
                msg = "Disabled " + job_id if job else "Job not found: " + job_id
            console.print(f"[dim]  {msg}[/dim]")
            return

    if sub == "add":
        if len(parts) < 5:
            console.print("[dim]  Usage: /cron add every <seconds> <message>[/dim]")
            console.print("[dim]         /cron add at <iso-datetime> <message>[/dim]")
            console.print("[dim]         /cron add expr <cron-expr> <message> [tz][/dim]")
            return
        mode = parts[2].lower()
        if mode == "every":
            try:
                seconds = int(parts[3])
            except ValueError:
                console.print("[dim]  Invalid seconds value.[/dim]")
                return
            message = " ".join(parts[4:]).strip()
            job = CRON.add_job(
                name=message[:40],
                schedule=CronSchedule(kind="every", every_ms=seconds * 1000),
                message=message,
                deliver=True,
                channel="cli",
                to="direct",
                session_key="manual_cron",
            )
            console.print(f"[dim]  Created job {job.id} ({job.name}).[/dim]")
            return
        if mode == "at":
            try:
                at_ms = int(datetime.fromisoformat(parts[3]).timestamp() * 1000)
            except ValueError:
                console.print("[dim]  Invalid ISO datetime.[/dim]")
                return
            message = " ".join(parts[4:]).strip()
            job = CRON.add_job(
                name=message[:40],
                schedule=CronSchedule(kind="at", at_ms=at_ms),
                message=message,
                deliver=True,
                channel="cli",
                to="direct",
                session_key="manual_cron",
                delete_after_run=True,
            )
            console.print(f"[dim]  Created job {job.id} ({job.name}).[/dim]")
            return
        if mode == "expr":
            if len(parts) < 5:
                console.print("[dim]  Usage: /cron add expr <cron-expr> <message> [tz][/dim]")
                return
            if _get_croniter() is None:
                console.print("[dim]  croniter is not installed, cron expressions are unavailable.[/dim]")
                return
            expr = parts[3]
            message = parts[4]
            tz = parts[5] if len(parts) >= 6 else None
            job = CRON.add_job(
                name=message[:40],
                schedule=CronSchedule(kind="cron", expr=expr, tz=tz),
                message=message,
                deliver=True,
                channel="cli",
                to="direct",
                session_key="manual_cron",
            )
            console.print(f"[dim]  Created job {job.id} ({job.name}).[/dim]")
            return

    console.print("[dim]  Unknown /cron usage. Subcommands: list, run, rm, on, off, add[/dim]")


def _render_history(history: list[dict]) -> None:
    """Pretty-print previous conversation after /resume, dimmed with a divider."""
    sync_console_width(console)

    def _is_visible(m: dict) -> bool:
        if m.get("role") not in ("user", "assistant"):
            return False
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            return False
        skip_prefixes = (
            "<background-results>",
            "<reminder>",
            "[System: Context auto-compressed",
            "[System: User was idle",
        )
        return not c.startswith(skip_prefixes)

    visible = [m for m in history if _is_visible(m)]
    if not visible:
        return

    shown = visible[-_REPLAY_TAIL:]
    omitted = len(visible) - len(shown)

    console.rule("[dim]session history[/dim]", style="dim")
    if omitted > 0:
        console.print(
            f"[dim italic]  \u2026 {omitted} earlier message(s) hidden \u2026[/dim italic]\n"
        )

    for msg in shown:
        body = msg["content"]
        if len(body) > 2000:
            body = body[:2000] + f"\n[\u2026 {len(body) - 2000} chars truncated \u2026]"
        # Escape Rich markup so user content like "[bold]" isn't interpreted.
        from rich.markup import escape
        body_esc = escape(body)
        if msg["role"] == "user":
            _render_history_input_box(body)
        else:
            indent = " " * ASSISTANT_LEFT_PADDING
            console.print("[bold #d78cff]Nova:[/bold #d78cff]")
            console.print(f"{indent}[dim]{body_esc}[/dim]", highlight=False, soft_wrap=True)
        console.print()

    console.rule(style="dim")
    console.print()


def _time_ago(dt) -> str:
    delta = time.time() - dt.timestamp()
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _format_timestamp(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _render_history_input_box(body: str) -> None:
    width = min(_HISTORY_BOX_WIDTH, content_width())
    inner_width = max(20, width - 4)
    title = " you "
    top_fill = "-" * max(1, width - len(title) - 2)
    console.print(f"[dim]+-{title}{top_fill}+[/dim]")
    lines: list[str] = []
    for raw_line in body.splitlines() or [""]:
        wrapped = textwrap.wrap(raw_line, width=inner_width) or [""]
        lines.extend(wrapped)
    for line in lines:
        safe = escape(line[:inner_width])
        console.print(f"[dim]| {safe.ljust(inner_width)} |[/dim]")
    console.print(f"[dim]+{'-' * (width - 2)}+[/dim]")


def _render_subagent_list() -> None:
    tasks = SUBAGENT.list_all()
    if not tasks:
        console.print("[dim]  No subagents.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Capability", no_wrap=True)
    table.add_column("Tools", no_wrap=True)
    table.add_column("Started", no_wrap=True)
    table.add_column("Description")

    for task in tasks:
        mode = "bg" if task.get("is_backgrounded", True) else "fg"
        table.add_row(
            task["task_id"],
            task["status"],
            mode,
            task.get("capability", "-"),
            str(task.get("tool_uses", 0)),
            _format_timestamp(task.get("started_at")),
            task.get("description", ""),
        )
    console.print(table)


def _print_subagent_blob(title: str, body: str) -> None:
    console.rule(f"[dim]{title}[/dim]", style="dim")
    console.print(escape(body) if body else "[dim](empty)[/dim]", highlight=False, soft_wrap=True)


def _render_subagent_detail(task: dict[str, object]) -> None:
    if task.get("error") and "task_id" not in task:
        console.print(f"[dim]  {task['error']}[/dim]")
        return

    status = str(task.get("status", "-"))
    mode = "background" if task.get("is_backgrounded", True) else "foreground"
    console.print(f"[cyan]{escape(str(task.get('task_id', '-')))}[/cyan] [dim]({escape(status)} / {mode})[/dim]")
    console.print(f"[dim]  Capability : {escape(str(task.get('capability', '-')))}[/dim]")
    console.print(f"[dim]  Description: {escape(str(task.get('description', '')))}[/dim]")
    console.print(f"[dim]  Started    : {_format_timestamp(task.get('started_at'))}[/dim]")
    console.print(f"[dim]  Finished   : {_format_timestamp(task.get('finished_at'))}[/dim]")
    console.print(f"[dim]  Tool uses  : {task.get('tool_uses', 0)}[/dim]")
    console.print(f"[dim]  Output     : {escape(str(task.get('output_file', '-')))}[/dim]")
    console.print(f"[dim]  Transcript : {escape(str(task.get('transcript_file', '-')))}[/dim]")
    if task.get("stop_requested"):
        console.print(f"[dim]  Stop req   : {escape(str(task.get('stop_reason') or 'requested'))}[/dim]")
    if task.get("error") and status not in {"running"}:
        console.print(f"[dim]  Error      : {escape(str(task.get('error')))}[/dim]")

    output_preview = str(task.get("output_preview", "") or "")
    transcript_preview = str(task.get("transcript_preview", "") or "")
    if output_preview:
        _print_subagent_blob("output preview", output_preview)
    if transcript_preview:
        _print_subagent_blob("transcript preview", transcript_preview)
    console.print()


def _resolve_session_summary(state: dict) -> str | None:
    """Prefer explicit session metadata, then fall back to compacted history."""
    metadata = state.get("metadata", {})
    summary = metadata.get("session_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return extract_session_summary(state.get("messages", []))


async def _pick_session_interactive(
    sessions: list[dict],
    *,
    current_session_key: str,
) -> dict | None:
    """Inline arrow-key session picker for bare /resume."""
    if not sessions:
        return None

    visible = sessions[:_SESSION_PICK_LIMIT]
    selected_index = next(
        (idx for idx, item in enumerate(visible) if item["key"] == current_session_key),
        0,
    )

    def _render_picker():
        fragments: list[tuple[str, str]] = [
            ("class:title", "Resume session\n"),
            ("class:hint", "Up/Down select, Enter resume, Esc cancel.\n\n"),
        ]
        for idx, item in enumerate(visible):
            ago = _time_ago(item["updated_at"])
            pointer = "> " if idx == selected_index else "  "
            key_style = "class:selected" if idx == selected_index else "class:item"
            meta_style = "class:selected-meta" if idx == selected_index else "class:meta"
            current = "  [current]" if item["key"] == current_session_key else ""
            fragments.append((key_style, f"{pointer}{item['key']}{current}\n"))
            fragments.append((meta_style, f"   {item['message_count']} msgs, {ago}\n"))
            fragments.append(("", "\n"))
        if len(sessions) > len(visible):
            fragments.append(
                ("class:hint", f"Showing latest {len(visible)} of {len(sessions)} sessions.\n")
            )
        return fragments

    body = FormattedTextControl(_render_picker, focusable=True, show_cursor=False)
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _move_up(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(visible)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _move_down(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(visible)
        event.app.invalidate()

    @kb.add("enter")
    def _accept(event) -> None:
        event.app.exit(result=visible[selected_index])

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    app = Application(
        layout=Layout(HSplit([Window(content=body, always_hide_cursor=True)])),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        style=Style.from_dict({
            "title": "bold",
            "hint": "ansibrightblack",
            "item": "",
            "meta": "ansibrightblack",
            "selected": "reverse bold",
            "selected-meta": "reverse",
        }),
    ) 

    with patch_stdout():
        return await app.run_async()


_NOVA_MARK = (
    ("bold #66e3ff", "  ███╗     ███╗   ██╗  ██████╗  ██╗   ██╗  █████╗    "),
    ("bold #48c9ff", "  ╚═███╗   ████╗  ██║ ██╔═══██╗ ██║   ██║ ██╔══██╗   "),
    ("bold #2aa7ff", "    ╚═███╗ ██╔██╗ ██║ ██║   ██║ ██║   ██║ ███████║   "),
    ("bold #7c7dff", "    ███╔═╝ ██║╚██╗██║ ██║   ██║ ╚██╗ ██╔╝ ██╔══██║   "),
    ("bold #b46cff", "  ███╔═╝   ██║ ╚████║ ╚██████╔╝  ╚████╔╝  ██║  ██║   "),
    ("bold #d86cff", "  ╚══╝     ╚═╝  ╚═══╝  ╚═════╝    ╚═══╝   ╚═╝  ╚═╝   "),
)


def _server_capability_lines(mcp_client) -> list[str]:
    if not mcp_client or not mcp_client.connected_servers:
        return ["[mcp] no MCP servers connected"]
    lines: list[str] = []
    for idx, server_name in enumerate(mcp_client.connected_servers):
        caps = mcp_client.server_capabilities.get(
            server_name,
            {"tools": [], "resources": [], "prompts": []},
        )
        tool_count = len(caps.get("tools", []))
        resource_count = len(caps.get("resources", []))
        prompt_count = len(caps.get("prompts", []))
        total = tool_count + resource_count + prompt_count
        if idx == 0:
            lines.append(f"[mcp] Connected '{server_name}': {total} capabilities")
        lines.append(
            f" {server_name}: {total} capabilities "
            f"({tool_count} tools, {resource_count} resources, {prompt_count} prompts)"
        )
    return lines


def _truncate_middle(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    keep = width - 3
    left = max(1, keep // 2)
    right = max(1, keep - left)
    return f"{text[:left]}...{text[-right:]}"


def _print_banner(heartbeat, *, mcp_client=None) -> None:
    width = sync_console_width(console)
    console.print()
    hb = heartbeat.status()
    side_width = max(24, width - BANNER_SIDE_RESERVE)
    status_lines = [
        ("", "Lightweight personal AI assistant"),
        ("Model", MODEL),
        ("Workspace", str(WORKDIR)),
        ("Runtime", str(RUNTIME_DIR)),
        (
            "Heartbeat:",
            f"{'running' if hb['running'] else 'stopped'}, every {hb['interval_s']}s, file={'yes' if hb['present'] else 'no'}",
        ),
    ]
    for idx, (style, line) in enumerate(_NOVA_MARK):
        label, value = status_lines[idx] if idx < len(status_lines) else ("", "")
        if label:
            label_text = f"{label} "
            value_width = max(1, side_width - len(label_text))
            truncated_val = _truncate_middle(value, value_width)
            if label == "Model":
                side = f"[dim]{escape(label_text)}[/dim][bold blue]{escape(truncated_val)}[/bold blue]"
            else:
                side = f"[dim]{escape(label_text)}[/dim][white]{escape(truncated_val)}[/white]"
        else:
            side = f"[white]{escape(_truncate_middle(value, side_width))}[/white]"
        console.print(f"[{style}]{line}[/]  {side}")
        
        # if label:
        #     label_text = f"{label} "
        #     value_width = max(1, side_width - len(label_text))
        #     side = f"[dim]{escape(label_text)}[/dim][white]{escape(_truncate_middle(value, value_width))}[/white]"
        # else:
        #     side = f"[white]{escape(_truncate_middle(value, side_width))}[/white]"
        # console.print(f"[{style}]{line}[/]  {side}")

    console.print()
    for idx, line in enumerate(_server_capability_lines(mcp_client)):
        if idx == 0:
            console.print(f"[bold white]{escape(line)}[/bold white]")
        else:
            escaped = escape(line)
            escaped = escaped.replace("capabilities", "[cyan]capabilities[/cyan]")
            escaped = escaped.replace("tools", "[cyan]tools[/cyan]")
            escaped = escaped.replace("resources", "[cyan]resources[/cyan]")
            escaped = escaped.replace("prompts", "[cyan]prompts[/cyan]")
            console.print(f"[dim]{escaped}[/dim]")

    console.print()
    console.print("[bold]Tips for getting started:[/bold]")
    console.print("1. Ask questions, edit files, or run commands.")
    console.print("2. Be specific for the best results.")
    console.print("3. [cyan]/help[/cyan] for more information.")


async def main():
    # --- Seed workspace templates (first run) ---
    seed_workspace_templates()
    # Skill loader is instantiated at import time; refresh after first-run seeding.
    SKILLS.reload()

    # --- One-shot dedup cleanup of accumulated memory files ---
    cleanup_memory_files_once()

    # --- Start fresh session (no picker) ---
    store = SessionStore(
        SESSION_DIR,
        workspace=WORKDIR,
        legacy_sessions_dir=LEGACY_SESSION_DIR,
    )
    session_key = f"session_{int(time.time())}"
    history: list[dict] = []
    session_summary: str | None = None
    interrupts = _InterruptController()
    interrupts.install()

    # --- MCP initialization (optional) ---
    mcp_client = await load_mcp(MCP_CONFIG_PATH)
    all_tools = list(TOOLS)
    all_handlers = dict(TOOL_HANDLERS)
    if mcp_client:
        all_tools.extend(mcp_client.tool_schemas)
        all_handlers.update(mcp_client.tool_handlers)
    set_permission_prompt_handler(_permission_prompt)
    set_ask_handler(_ask_user_handler)

    async def _run_background_turn(
        prompt: str,
        *,
        run_channel: str,
        run_chat_id: str,
        run_session_key: str,
        emit_output: bool,
        assistant_label: str,
    ) -> str | None:
        state = store.load_state(run_session_key)
        run_history = list(state["messages"])
        run_summary = _resolve_session_summary(state)
        user_msg = {"role": "user", "content": prompt}
        store.update_metadata(run_session_key, pending_user_turn=True)
        run_history.append(user_msg)
        store.append(run_session_key, user_msg)
        return await agent_loop(
            messages=run_history,
            system=build_system_prompt(),
            tools=all_tools,
            tool_handlers=all_handlers,
            todo_mgr=TODO,
            bg_mgr=BG,
            session_store=store,
            session_key=run_session_key,
            channel=run_channel,
            chat_id=run_chat_id,
            session_summary=run_summary,
            emit_output=emit_output,
            assistant_label=assistant_label,
        )

    async def _handle_cron_job(job) -> str | None:
        if job.payload.kind == "system_event" and job.id == _MEMORY_JOB_ID:
            await consolidate_memory(
                [],
                store=_MEMORY,
                emit_output=False,
                session_key=job.payload.session_key,
                channel=job.payload.channel or "cron",
                chat_id=job.payload.to or "cron",
            )
            return None
        await _interactive_notice(f"[cron] running {job.name} ({job.id})")
        response = await _run_background_turn(
            job.payload.message,
            run_channel=job.payload.channel or "cron",
            run_chat_id=job.payload.to or "cron",
            run_session_key=job.payload.session_key or f"cron_{job.id}",
            emit_output=False,
            assistant_label=f"cron:{job.name}",
        )
        if job.payload.deliver and response and not response.isspace():
            await _interactive_notice(f"[cron] completed {job.name}")
            await _interactive_response(f"cron:{job.name}", response)
        return response

    async def _heartbeat_execute(tasks: str) -> str | None:
        await _interactive_notice("[heartbeat] active task detected")
        return await _run_background_turn(
            tasks,
            run_channel="heartbeat",
            run_chat_id="heartbeat",
            run_session_key="heartbeat",
            emit_output=False,
            assistant_label="heartbeat",
        )

    async def _heartbeat_notify(response: str) -> None:
        await _interactive_notice("[heartbeat] result")
        await _interactive_response("heartbeat", response)

    heartbeat = HeartbeatService(
        WORKDIR,
        on_execute=_heartbeat_execute,
        on_notify=_heartbeat_notify,
        interval_s=HEARTBEAT_INTERVAL_SECONDS,
    )
    CRON.set_handler(_handle_cron_job)
    CRON.register_system_job(CronJob(
        id=_MEMORY_JOB_ID,
        name="memory_consolidation",
        schedule=CronSchedule(kind="every", every_ms=MEMORY_CONSOLIDATION_INTERVAL_SECONDS * 1000),
        payload=CronPayload(kind="system_event"),
    ))

    # AutoCompact periodic scan (runs alongside heartbeat interval)
    _AUTOCOMPACT_JOB_ID = "autocompact_scan"

    async def _autocompact_cron_handler(job):
        ac = get_autocompact(store)
        if ac is not None:
            ac.check_expired(
                lambda coro: asyncio.create_task(coro),
                active_session_keys=set(),
            )
        return None

    if IDLE_COMPACT_MINUTES > 0:
        original_handler = CRON._handler

        async def _combined_handler(job):
            if job.id == _AUTOCOMPACT_JOB_ID:
                return await _autocompact_cron_handler(job)
            return await original_handler(job)

        CRON._handler = _combined_handler
        CRON.register_system_job(CronJob(
            id=_AUTOCOMPACT_JOB_ID,
            name="autocompact_idle_scan",
            schedule=CronSchedule(kind="every", every_ms=HEARTBEAT_INTERVAL_SECONDS * 1000),
            payload=CronPayload(kind="system_event"),
        ))

    await CRON.start()
    await heartbeat.start()

    # --- Welcome banner ---
    _print_banner(
        heartbeat,
        mcp_client=mcp_client,
    )

    try:
        while True:
            try:
                query = await _ask_user(history)
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("[dim]  Input cancelled. Use exit or /exit to quit.[/dim]")
                continue

            if not query:
                continue
            if query.lower() in ("exit", "quit", "/exit", "/quit"):
                break

            # ---- REPL commands ----
            if query == "/help":
                console.print(_HELP_TEXT)
                continue

            if query == "/status":
                console.print(f"[dim]  Session : {session_key}[/dim]")
                console.print(f"[dim]  Messages: {len(history)}[/dim]")
                console.print(f"[dim]  Model   : {MODEL}[/dim]")
                console.print(f"[dim]  Summary : {'yes' if session_summary else 'none'}[/dim]")
                for line in _format_cron_snapshot():
                    console.print(f"[dim]{line}[/dim]")
                hb = heartbeat.status()
                console.print(
                    f"[dim]  Heartbeat: {'running' if hb['running'] else 'stopped'}, "
                    f"every {hb['interval_s']}s, file={'yes' if hb['present'] else 'no'}[/dim]"
                )
                if hb.get("last_action"):
                    console.print(
                        f"[dim]  Heartbeat last: {hb['last_action']} "
                        f"({hb.get('last_reason') or 'n/a'})[/dim]"
                    )
                for line in render_memory_status_lines(
                    get_memory_operator_status(store=_MEMORY),
                    detailed=False,
                ):
                    console.print(f"[dim]  {line}[/dim]")
                if mcp_client and mcp_client.connected_servers:
                    console.print(
                        f"[dim]  MCP     : {len(mcp_client.connected_servers)} server(s), "
                        f"{len(mcp_client.tool_schemas)} capabilities[/dim]"
                    )
                    for line in mcp_client.startup_summary_lines():
                        console.print(f"[dim] {line}[/dim]")
                else:
                    console.print("[dim]  MCP     : none[/dim]")
                continue

            if query == "/mcp":
                if mcp_client and mcp_client.connected_servers:
                    _render_mcp_details(mcp_client)
                else:
                    console.print("[dim]  No MCP servers connected.[/dim]")
                continue

            if query.startswith("/cron"):
                await _handle_cron_command(query)
                continue

            if query == "/heartbeat":
                console.print("[dim]  Triggering heartbeat...[/dim]")
                result = await heartbeat.trigger_now()
                if result is None:
                    console.print("[dim]  Heartbeat skipped.[/dim]")
                continue

            if query == "/new":
                session_key = f"session_{int(time.time())}"
                history.clear()
                session_summary = None
                console.clear()
                _print_banner(
                    heartbeat,
                    mcp_client=mcp_client,
                )
                continue

            if query == "/sessions":
                sessions = store.list_sessions()
                if not sessions:
                    console.print("[dim]  No saved sessions.[/dim]")
                else:
                    console.print(f"[dim]  Workspace sessions: {WORKDIR}[/dim]")
                    for i, s in enumerate(sessions[:15], 1):
                        ago = _time_ago(s["updated_at"])
                        cur = " [bold cyan]<-[/bold cyan]" if s["key"] == session_key else ""
                        console.print(
                            f"  [dim]{i:>2}.[/dim] {s['key']}  "
                            f"[dim]({s['message_count']} msgs, {ago})[/dim]{cur}"
                        )
                continue

            if query.startswith("/resume"):
                arg = query[len("/resume"):].strip()
                sessions = store.list_sessions()
                target = None
                if not arg:
                    if not sessions:
                        console.print("[dim]  No saved sessions.[/dim]")
                        continue
                    target = await _pick_session_interactive(
                        sessions,
                        current_session_key=session_key,
                    )
                    if target is None:
                        console.print("[dim]  Resume cancelled.[/dim]")
                        continue
                else:
                    try:
                        idx = int(arg) - 1
                        if 0 <= idx < len(sessions):
                            target = sessions[idx]
                    except ValueError:
                        for s in sessions:
                            if s["key"] == arg:
                                target = s
                                break
                if target:
                    session_key = target["key"]
                    state = store.load_state(session_key)
                    history[:] = state["messages"]
                    session_summary = _resolve_session_summary(state)

                    idle_minutes = (time.time() - target["updated_at"].timestamp()) / 60
                    if history and idle_minutes > 60 and len(history) > 10:
                        console.print(
                            f"[dim]  Idle {int(idle_minutes)}m — compressing older history...[/dim]"
                        )
                        history[:] = await auto_compact(
                            history,
                            is_idle=True,
                            idle_minutes=idle_minutes,
                            memory_store=_MEMORY,
                        )
                        store.save_all(session_key, history)
                        session_summary = extract_session_summary(history)
                        store.update_metadata(session_key, session_summary=session_summary or "")

                    if history:
                        _render_history(history)
                    if session_summary:
                        console.print("[dim]  Resume summary loaded into runtime context.[/dim]")
                    console.print(
                        f"[dim]  Resumed '{session_key}' ({len(history)} messages).[/dim]"
                    )
                else:
                    console.print("[dim]  Session not found. Use /resume (bare) for picker, /resume <#> for index, or /resume <key>.[/dim]")
                continue

            if query == "/compact":
                if history:
                    console.print("[dim]  Compressing...[/dim]")
                    history[:] = await auto_compact(history, is_idle=False, memory_store=_MEMORY)
                    store.save_all(session_key, history)
                    session_summary = extract_session_summary(history)
                    store.update_metadata(session_key, session_summary=session_summary or "")
                    console.print("[dim]  Done.[/dim]")
                continue

            if query.startswith("/memory"):
                parts = shlex.split(query)
                if len(parts) > 1 and parts[1].lower() == "status":
                    for line in render_memory_status_lines(
                        get_memory_operator_status(store=_MEMORY),
                        detailed=True,
                    ):
                        console.print(f"[dim]  {line}[/dim]")
                    continue
                if len(parts) > 1 and parts[1].lower() == "export":
                    backend = create_memory_backend(_MEMORY)
                    status = backend.status()
                    if status.backend not in {"mem0", "hybrid"}:
                        console.print("[dim]  Snapshot export requires NOVA_MEMORY_BACKEND=mem0 or hybrid.[/dim]")
                        continue
                    if not status.healthy:
                        console.print(
                            f"[dim red]  Snapshot export failed: {status.last_error or 'memory backend is unhealthy'}[/dim red]"
                        )
                        continue
                    result = export_memory_snapshot(
                        backend=backend,
                        store=_MEMORY,
                        session_key=session_key,
                    )
                    if result.ok:
                        console.print(
                            f"[dim]  Exported {result.count} memory item(s) to {result.path}.[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim red]  Snapshot export failed: {result.error or 'unknown error'}[/dim red]"
                        )
                    continue
                if len(parts) > 1 and parts[1].lower() == "migrate":
                    result = run_legacy_memory_migration(
                        store=_MEMORY,
                        session_key=session_key,
                    )
                    if result.ok and result.skipped:
                        console.print("[dim]  Legacy memory migration was already completed.[/dim]")
                    elif result.ok:
                        console.print(
                            f"[dim]  Migrated {result.migrated} legacy memory batch(es) into Mem0.[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim red]  Memory migration failed: {result.error or 'unknown error'}[/dim red]"
                        )
                    continue
                if len(parts) > 1:
                    console.print("[dim]  Usage: /memory | /memory status | /memory export | /memory migrate[/dim]")
                    continue
                console.print("[dim]  Running memory consolidation...[/dim]")
                wrote = await consolidate_memory(
                    history,
                    store=_MEMORY,
                    session_key=session_key,
                    channel="cli",
                    chat_id="direct",
                    force=True,
                    sync_mem0=True,
                )
                if wrote:
                    console.print("[dim]  Memory consolidation completed.[/dim]")
                else:
                    console.print("[dim]  No durable memory was written this time.[/dim]")
                continue

            if query.startswith("/dream-log"):
                _handle_dream_log_command(query)
                continue

            if query.startswith("/dream-restore"):
                _handle_dream_restore_command(query)
                continue

            if query == "/tasks":
                console.print(TASK_MGR.list_all())
                continue

            if query.startswith("/bg"):
                parts = shlex.split(query)
                if len(parts) == 1:
                    console.print(BG.check())
                    continue
                if len(parts) == 2:
                    console.print(BG.check(parts[1]))
                    continue
                if len(parts) >= 3 and parts[1].lower() == "output":
                    console.print(
                        json.dumps(
                            BG.task_output(parts[2], block=False, timeout_ms=0),
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    continue
                console.print("[dim]  Usage: /bg | /bg <task_id> | /bg output <task_id>[/dim]")
                continue

            if query.startswith("/subagents"):
                parts = shlex.split(query)
                if len(parts) == 1:
                    _render_subagent_list()
                    continue

                if len(parts) == 2:
                    _render_subagent_detail(SUBAGENT.status(parts[1]))
                    continue

                sub = parts[1].lower()
                if sub in {"output", "transcript"}:
                    if len(parts) < 3:
                        console.print(f"[dim]  Usage: /subagents {sub} <task_id>[/dim]")
                        continue
                    detail = SUBAGENT.detail(parts[2])
                    if detail.get("error"):
                        console.print(f"[dim]  {detail['error']}[/dim]")
                        continue
                    body = detail.get(sub, "")
                    _print_subagent_blob(f"{sub} {parts[2]}", str(body or ""))
                    console.print()
                    continue

                if sub in {"fg", "foreground", "wait"}:
                    if len(parts) < 3:
                        console.print("[dim]  Usage: /subagents fg <task_id>[/dim]")
                        continue
                    result = await SUBAGENT.wait(
                        parts[2],
                        timeout_ms=None,
                        foreground=True,
                        include_output=True,
                    )
                    if result.get("retrieval_status") == "not_found":
                        console.print(f"[dim]  Unknown task_id: {parts[2]}[/dim]")
                        continue
                    task = result.get("task") or {}
                    console.print(f"[dim]  Foreground wait finished: {task.get('status', 'unknown')}[/dim]")
                    _render_subagent_detail(SUBAGENT.status(parts[2]))
                    continue

                if sub in {"bg", "background"}:
                    if len(parts) < 3:
                        console.print("[dim]  Usage: /subagents bg <task_id>[/dim]")
                        continue
                    result = SUBAGENT.set_backgrounded(parts[2], True)
                    if result.get("error"):
                        console.print(f"[dim]  {result['error']}[/dim]")
                    else:
                        console.print(f"[dim]  {parts[2]} moved to background.[/dim]")
                    continue

                if sub in {"stop", "interrupt", "kill"}:
                    if len(parts) < 3:
                        console.print("[dim]  Usage: /subagents stop <task_id> [reason][/dim]")
                        continue
                    reason = " ".join(parts[3:]).strip() or "stopped by user"
                    result = SUBAGENT.stop(parts[2], reason=reason)
                    if result.get("error"):
                        console.print(f"[dim]  {result['error']}[/dim]")
                    else:
                        console.print(f"[dim]  stop requested for {parts[2]}.[/dim]")
                    continue

                console.print(
                    "[dim]  Unknown subcommand. Try: /subagents output|transcript|fg|bg|stop <id>[/dim]"
                )
                continue

            if query == "/permissions":
                from nova.tools.registry import PERMISSIONS
                console.print(json.dumps(PERMISSIONS.list_rules(), indent=2, ensure_ascii=False))
                continue

            if query.startswith("/"):
                console.print(f"[dim]  Unknown command: {query}[/dim]")
                console.print("[dim]  Commands: /new /sessions /resume /compact /memory /dream-log /dream-restore /cron /heartbeat /mcp /tasks /bg /subagents /permissions /status /help[/dim]")
                continue

            # ---- Normal message ----
            system = build_system_prompt()
            user_msg = {"role": "user", "content": query}
            store.update_metadata(session_key, pending_user_turn=True)
            history.append(user_msg)
            store.append(session_key, user_msg)

            try:
                await interrupts.run_interruptible(
                    agent_loop(
                        messages=history,
                        system=system,
                        tools=all_tools,
                        tool_handlers=all_handlers,
                        todo_mgr=TODO,
                        bg_mgr=BG,
                        session_store=store,
                        session_key=session_key,
                        channel="cli",
                        chat_id="direct",
                        session_summary=session_summary,
                    )
                )
            except (_GenerationInterrupted, KeyboardInterrupt, asyncio.CancelledError):
                _rollback_interrupted_turn(store, session_key, history)
                console.print(
                    "[dim yellow]  Generation interrupted. Session is still active.[/dim yellow]"
                )
            except Exception as e:
                console.print(f"[red]  Error in agent loop: {e}[/red]")
                console.print(
                    "[dim yellow]  Tip: use /compact to shrink context or /new to start fresh.[/dim yellow]"
                )

    finally:
        interrupts.restore()
        set_permission_prompt_handler(None)
        set_ask_handler(None)
        await heartbeat.stop()
        await CRON.stop()
        if mcp_client:
            await mcp_client.close()
        console.print("\n[dim]Goodbye![/dim]")
