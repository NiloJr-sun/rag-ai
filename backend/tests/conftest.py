"""Keep tests independent of the developer's shell environment.

Several tests assert on the built-in defaults, so an exported variable -- or
a line in the repo-root .env, which app.config now loads -- would otherwise
make them pass in CI and fail locally.
"""

from __future__ import annotations

import pytest

CONFIG_VARS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_CHAT_MODEL",
    "RETRIEVAL_TOP_K",
    "API_HOST",
    "API_PORT",
)


@pytest.fixture(autouse=True)
def _clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
