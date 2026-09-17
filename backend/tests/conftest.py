"""Keep tests independent of the developer's shell environment.

Several tests assert on the built-in defaults, so an exported OLLAMA_*
variable would otherwise make them pass in CI and fail locally.
"""

from __future__ import annotations

import pytest

OLLAMA_VARS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_CHAT_MODEL",
)


@pytest.fixture(autouse=True)
def _clear_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OLLAMA_VARS:
        monkeypatch.delenv(name, raising=False)
