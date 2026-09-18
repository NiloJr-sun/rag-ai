"""Every setting the application reads, in one place.

Values resolve in this order:

1. a real environment variable
2. a line in the repo-root .env (loaded here, once)
3. the default in this module

load_dotenv never overrides an existing environment variable, so an export
still wins. Reading happens per call rather than being cached at import, so
tests can monkeypatch the environment and so a long-running process picks up
a change without a restart.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

# --- Ollama (T1.5) ---
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "qwen2"

# --- retrieval (T1.9, T2.8) ---
DEFAULT_TOP_K = 5

# --- API (T2.1) ---
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000


class ConfigError(ValueError):
    """A setting is present but unusable."""


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def top_k() -> int:
    """How many chunks retrieval feeds to the model.

    Worth tuning rather than guessing: too few and the model has no context to
    answer from, too many and the prompt fills with irrelevant text (T2.8).
    """
    return _int("RETRIEVAL_TOP_K", DEFAULT_TOP_K)


def api_host() -> str:
    return os.environ.get("API_HOST", DEFAULT_API_HOST)


def api_port() -> int:
    return _int("API_PORT", DEFAULT_API_PORT)


def database_url() -> str | None:
    """Postgres connection string, or None when it is not configured."""
    return os.environ.get("DATABASE_URL") or None


def describe() -> dict[str, str]:
    """Effective settings, safe to expose: no secret values.

    DATABASE_URL carries a password, so only whether it is set is reported.
    """
    return {
        "ollama_base_url": ollama_base_url(),
        "embedding_model": embedding_model(),
        "chat_model": chat_model(),
        "top_k": str(top_k()),
        "database_configured": str(database_url() is not None).lower(),
    }
