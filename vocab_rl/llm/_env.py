"""Helper: load .env from the project root, once per process.

Imported by every llm/ entrypoint that needs ANTHROPIC_API_KEY. python-dotenv
is optional — if it's not installed we silently no-op and rely on the env var
being set externally.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env() -> None:
    """Idempotent. Loads vocab_rl/.env and project_root/.env if present."""
    global _LOADED
    if _LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        return
    here = Path(__file__).resolve()
    # try vocab_rl/.env first, then project_root/.env
    for candidate in (here.parent.parent / ".env", here.parent.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
    _LOADED = True


def require_api_key() -> str:
    """Loads .env and asserts the key is set. Raises a clear error if not."""
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either:\n"
            "  (a) create a .env file in the project root with "
            "`ANTHROPIC_API_KEY=sk-ant-...`, or\n"
            "  (b) export it in your shell: `export ANTHROPIC_API_KEY=sk-ant-...`."
        )
    return key
