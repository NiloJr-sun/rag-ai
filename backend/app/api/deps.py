"""Shared request dependencies (T2.2).

A connection and an HTTP client are opened per request and closed with it.
That is deliberately simple rather than efficient: connection pooling and
moving ingestion off the request thread are T6.3's job, and doing either now
would be guessing at a shape the background-worker ticket will change.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import psycopg

from app.rag.generation import DEFAULT_TIMEOUT_SECONDS
from app.storage import supabase


def get_connection() -> Iterator[psycopg.Connection]:
    with supabase.connect() as conn:
        yield conn


def get_http_client() -> Iterator[httpx.Client]:
    # One client for the whole request, so retrieval's embedding call and the
    # chat call that follows reuse a single connection to Ollama.
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        yield client
