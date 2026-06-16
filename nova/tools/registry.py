"""
nova/tools/registry.py - Global singleton instances, tool schema list,
and tool handler dispatch dict.
"""

from __future__ import annotations

from nova.background.manager import BackgroundManager
from nova.config import CRON_STORE_PATH, LEGACY_SKILLS_DIR, PERMISSIONS_FILE, SKILLS_DIR
from nova.cron.service import CronService
from nova.permissions import PermissionManager
from nova.skills.loader import SkillLoader
from nova.subagent.runner import SubagentRunner
from nova.tasks.manager import TaskManager
from nova.tasks.todo import TodoManager

TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR, extra_workspace_dirs=[LEGACY_SKILLS_DIR] if LEGACY_SKILLS_DIR.exists() else None)
TASK_MGR = TaskManager()
BG = BackgroundManager()
SUBAGENT = SubagentRunner()
CRON = CronService(CRON_STORE_PATH)
PERMISSIONS = PermissionManager(PERMISSIONS_FILE)

TOOL_HANDLERS: dict[str, object] = {}
TOOLS: list[dict] = []
_TOOL_INSTANCES: dict[str, object] = {}


def register_tool(tool_instance) -> None:
    """Register a BaseTool instance."""
    TOOLS.append(tool_instance.to_openai())
    TOOL_HANDLERS[tool_instance.name] = tool_instance.execute
    _TOOL_INSTANCES[tool_instance.name] = tool_instance


def get_tool_instance(name: str):
    """Return a registered tool instance by name."""
    return _TOOL_INSTANCES.get(name)


def set_permission_prompt_handler(handler) -> None:
    """Register an async permission prompt handler for interactive sessions."""
    PERMISSIONS.set_prompt_handler(handler)


def prepare_call(name: str, params: dict) -> tuple[object | None, dict, str | None]:
    """Resolve a tool and validate its parameters before execution."""
    if not isinstance(params, dict):
        return None, params, (
            f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}"
        )

    tool = _TOOL_INSTANCES.get(name)
    if tool is None:
        return None, params, f"Error: Tool '{name}' not found"

    cast_params = tool.cast_params(params)
    errors = tool.validate_params(cast_params)
    if errors:
        return tool, cast_params, f"Error: Invalid parameters for tool '{name}': {'; '.join(errors)}"
    return tool, cast_params, None


async def execute_registered_tool(name: str, params: dict) -> object:
    """Execute a registered tool with parameter preparation/validation."""
    tool, cast_params, error = prepare_call(name, params)
    if error:
        return error
    assert tool is not None
    result = tool.execute(**cast_params)
    if hasattr(result, "__await__"):
        return await result
    return result


def set_tool_runtime_context(
    *,
    channel: str = "cli",
    chat_id: str = "direct",
    session_key: str | None = None,
) -> None:
    """Push per-turn runtime context into tools that need it."""
    for tool in _TOOL_INSTANCES.values():
        setter = getattr(tool, "set_runtime_context", None)
        if callable(setter):
            setter(channel=channel, chat_id=chat_id, session_key=session_key)


def init_builtin_tools() -> None:
    from nova.tools.builtin.cron import CronTool
    from nova.tools.builtin.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from nova.tools.builtin.shell import BashTool
    from nova.tools.builtin.tasks import (
        ClaimTaskTool,
        TaskCreateTool,
        TaskGetTool,
        TaskListTool,
        TaskUpdateTool,
    )
    from nova.tools.builtin.background import (
        BackgroundRunTool,
        CheckBackgroundTool,
        TaskOutputTool,
    )
    from nova.tools.builtin.mem0_tools import (
        DeleteMemoryTool,
        GetMemoriesTool,
        GetMemoryTool,
        SearchMemoriesTool,
    )
    from nova.tools.builtin.skills import LoadSkillTool
    from nova.tools.builtin.ask import AskUserTool
    from nova.tools.builtin.subagent import (
        CheckSubagentTool,
        ControlSubagentTool,
        ListSubagentsTool,
        TaskTool,
        WaitSubagentTool,
    )
    from nova.tools.builtin.todo import CompressTool, TodoWriteTool

    for tool in [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        BashTool(),
        TaskCreateTool(),
        TaskGetTool(),
        TaskUpdateTool(),
        TaskListTool(),
        ClaimTaskTool(),
        TaskTool(),
        TodoWriteTool(),
        LoadSkillTool(),
        BackgroundRunTool(),
        CheckBackgroundTool(),
        CheckSubagentTool(),
        ListSubagentsTool(),
        ControlSubagentTool(),
        WaitSubagentTool(),
        TaskOutputTool(),
        SearchMemoriesTool(),
        GetMemoriesTool(),
        GetMemoryTool(),
        DeleteMemoryTool(),
        AskUserTool(),
        CompressTool(),
        CronTool(CRON),
    ]:
        register_tool(tool)


init_builtin_tools()
