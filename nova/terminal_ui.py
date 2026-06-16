"""Shared terminal layout helpers for Nova CLI rendering."""

from __future__ import annotations

import shutil

from rich.console import Console

MAX_CONTENT_WIDTH = 104
MIN_CONTENT_WIDTH = 24
ASSISTANT_LEFT_PADDING = 4
BANNER_SIDE_RESERVE = 54


def terminal_width(*, fallback: int = 80) -> int:
    """Return the live terminal width with a sane fallback."""
    try:
        return shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except Exception:
        return fallback


def content_width(
    *,
    max_width: int = MAX_CONTENT_WIDTH,
    min_width: int = MIN_CONTENT_WIDTH,
) -> int:
    """Clamp the render width between the live terminal width and our max width."""
    live_width = terminal_width(fallback=max_width)
    return max(min_width, min(max_width, max(min_width, live_width)))


def sync_console_width(
    console: Console,
    *,
    max_width: int = MAX_CONTENT_WIDTH,
    min_width: int = MIN_CONTENT_WIDTH,
) -> int:
    """Update a Rich console so wrapping respects our width cap."""
    width = content_width(max_width=max_width, min_width=min_width)
    console.width = width
    return width
