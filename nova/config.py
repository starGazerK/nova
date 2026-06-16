"""
nova/config.py - Global configuration and constants.

All environment variables, paths, thresholds are defined here.
LLM calls go through the provider abstraction layer
(nova.providers), which wraps litellm underneath.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required Nova runtime configuration is missing."""


WORKDIR = Path.cwd()
WORKDIR_ENV = WORKDIR / ".env"
if WORKDIR_ENV.exists():
    load_dotenv(dotenv_path=WORKDIR_ENV, override=True)

# --- LLM settings ---
# litellm uses provider-prefixed model names, e.g.:
#   anthropic/claude-3-5-sonnet  openai/gpt-4o  ollama/llama3  deepseek/deepseek-chat
try:
    MODEL = os.environ["MODEL_ID"]
    API_KEY = os.environ["API_KEY"]
except KeyError as exc:
    missing = exc.args[0]
    env_hint = str(WORKDIR_ENV)
    raise ConfigError(
        f"Missing required config: {missing}. "
        f"Set it in the environment or create {env_hint} with MODEL_ID and API_KEY."
    ) from exc
API_BASE = os.getenv("API_BASE")  # optional, for proxies or custom endpoints


def _is_gpt5_model(model: str) -> bool:
    return "gpt-5" in model.lower()


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_DEFAULT_MODEL_TEMPERATURE = 1.0 if _is_gpt5_model(MODEL) else 0.7
MODEL_TEMPERATURE = _parse_float_env("MODEL_TEMPERATURE", _DEFAULT_MODEL_TEMPERATURE)

# --- Workspace paths ---
TASKS_DIR = WORKDIR / ".tasks"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
RUNTIME_DIR = WORKDIR / ".nova"
SKILLS_DIR = RUNTIME_DIR / "skills"
MEMORY_DIR = RUNTIME_DIR / "memory"
CRON_DIR = RUNTIME_DIR / "cron"
SESSION_DIR = RUNTIME_DIR / "sessions"
BACKGROUND_DIR = RUNTIME_DIR / "background"
SUBAGENT_DIR = RUNTIME_DIR / "subagents"
PERMISSIONS_FILE = RUNTIME_DIR / "permissions.json"
LEGACY_SESSION_DIR = Path.home() / ".nova" / "sessions"
CRON_STORE_PATH = CRON_DIR / "jobs.json"
MCP_CONFIG_PATH = RUNTIME_DIR / "mcp_servers.json"
TASK_HOOKS_PATH = RUNTIME_DIR / "task_hooks.json"
AGENTS_MD_PATH = RUNTIME_DIR / "AGENTS.md"
SOUL_MD_PATH = RUNTIME_DIR / "SOUL.md"
USER_MD_PATH = RUNTIME_DIR / "USER.md"
TOOLS_MD_PATH = RUNTIME_DIR / "TOOLS.md"
HEARTBEAT_MD_PATH = RUNTIME_DIR / "HEARTBEAT.md"
LEGACY_SKILLS_DIR = WORKDIR / "skills"

# --- Tuning constants ---
TOKEN_THRESHOLD = 100_000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))
MEMORY_CONSOLIDATION_INTERVAL_SECONDS = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", "21600"))
IDLE_COMPACT_MINUTES = int(os.getenv("IDLE_COMPACT_MINUTES", "0"))

# --- Memory backend settings ---
# Values are loaded from the workspace `.env` before this module is evaluated.
# Keep Mem0's LLM/embedder provider settings separate from Nova's chat LLM:
# memory extraction and embeddings may need different models and endpoints.
NOVA_MEMORY_BACKEND = os.getenv("NOVA_MEMORY_BACKEND", "markdown").strip().lower()
NOVA_USER_ID = os.getenv("NOVA_USER_ID", "local-user").strip() or "local-user"
NOVA_MEMORY_SEARCH_LIMIT = int(os.getenv("NOVA_MEMORY_SEARCH_LIMIT", "6"))
NOVA_MEMORY_SEARCH_TIMEOUT_MS = int(os.getenv("NOVA_MEMORY_SEARCH_TIMEOUT_MS", "1200"))
NOVA_MEMORY_WRITE_MODE = os.getenv("NOVA_MEMORY_WRITE_MODE", "async").strip().lower()
MEM0_QDRANT_HOST = os.getenv("MEM0_QDRANT_HOST", "localhost").strip() or "localhost"
MEM0_QDRANT_PORT = int(os.getenv("MEM0_QDRANT_PORT", "6335"))
MEM0_COLLECTION = os.getenv("MEM0_COLLECTION", "nova_memories").strip() or "nova_memories"
MEM0_ENABLE_GRAPH = os.getenv("MEM0_ENABLE_GRAPH", "false").strip().lower() in {"1", "true", "yes", "on"}
MEM0_SNAPSHOT_LIMIT = int(os.getenv("MEM0_SNAPSHOT_LIMIT", "80"))
MEM0_RUNTIME_DIR = RUNTIME_DIR / "mem0"
MEM0_LLM_PROVIDER = os.getenv("MEM0_LLM_PROVIDER", "").strip()
MEM0_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", "").strip()
MEM0_LLM_API_KEY = os.getenv("MEM0_LLM_API_KEY", "").strip()
MEM0_LLM_BASE_URL = os.getenv("MEM0_LLM_BASE_URL", "").strip()
MEM0_EMBEDDER_PROVIDER = os.getenv("MEM0_EMBEDDER_PROVIDER", "").strip()
MEM0_EMBEDDER_MODEL = os.getenv("MEM0_EMBEDDER_MODEL", "").strip()
MEM0_EMBEDDER_API_KEY = os.getenv("MEM0_EMBEDDER_API_KEY", "").strip()
MEM0_EMBEDDER_BASE_URL = os.getenv("MEM0_EMBEDDER_BASE_URL", "").strip()
MEM0_EMBEDDER_DIMS = os.getenv("MEM0_EMBEDDER_DIMS", "").strip()

# --- Provider singleton ---
_PROVIDER = None


def create_provider():
    """Create and return the shared LLMProvider instance (lazy singleton)."""
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    from nova.providers.litellm_provider import LiteLLMProvider
    _PROVIDER = LiteLLMProvider(
        api_key=API_KEY,
        model=MODEL,
        api_base=API_BASE,
        temperature=MODEL_TEMPERATURE,
    )
    return _PROVIDER
