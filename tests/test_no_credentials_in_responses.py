"""No screen may serve a credential, and this checks by reading what the screens serve.

The console is public. It renders from a `Settings` object that holds `database_url` — a live Neon
DSN with a password in it on the deployed instance — and `llm_api_key`. For a while the whole object
was in the template context under `settings`, used by no template, which put both one
`{{ settings.database_url }}` away from a public page for anyone adding a debug line.

A grep over the templates would not be a guard. It would pass on the day someone renders the value
through a variable, a filter or an error handler, and it would pass on a stack trace. So this puts
sentinel values in the environment, asks for every screen, and reads the bytes that come back.

It runs with no database on purpose. `/evidence`, `/failures` and `/provenance` read `artifacts/`
and need none, `/` degrades, and `/healthz` reports 503 — and a degraded render is exactly where a
connection string is most likely to be echoed in an error message.

**Verified by planting the leak.** Putting `Settings` back in the context and adding
`<!-- debug: {{ settings.database_url }} -->` to `base.html` fails four of these: the three
artifact-backed screens and the context check. It does **not** fail the `/` case, because with an
unreachable database that route does not get as far as rendering the layout — which is worth
knowing rather than glossing, and is why the screens are parametrised instead of being one request.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from parts_answer_gate.api.app import app
from parts_answer_gate.config import get_settings

#: Shaped like the real thing — scheme, user, password, host, query — so that a partial echo is
#: caught too. It points at nothing; nothing in this file opens a connection.
SENTINEL_DSN = (
    "postgresql://sentineluser:sentinelpassword@sentinel.example.invalid/db?sslmode=require"
)
SENTINEL_KEY = "sk-sentinel-model-key-must-never-be-served"

#: Every fragment whose appearance would matter. The password and the key on their own, because a
#: template that rendered only `.password` would still have leaked it.
FORBIDDEN = (
    SENTINEL_DSN,
    "sentinelpassword",
    "sentineluser",
    "sentinel.example.invalid",
    SENTINEL_KEY,
)

SCREENS = ("/", "/evidence", "/failures", "/provenance")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PAG_DATABASE_URL", SENTINEL_DSN)
    monkeypatch.setenv("PAG_LLM_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("PAG_ENVIRONMENT", "test")
    get_settings.cache_clear()
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("path", SCREENS)
def test_no_screen_serves_a_credential(client: TestClient, path: str) -> None:
    response = client.get(path)
    body = response.text
    for secret in FORBIDDEN:
        assert secret not in body, f"{path} served {secret!r}"


def test_the_health_endpoints_report_state_without_reporting_the_dsn(client: TestClient) -> None:
    """`/healthz` names the database's condition. It must not name the database."""
    for path in ("/livez", "/healthz"):
        body = client.get(path).text
        for secret in FORBIDDEN:
            assert secret not in body, f"{path} served {secret!r}"


def test_an_error_response_does_not_echo_the_connection_string(client: TestClient) -> None:
    """The likeliest leak is not a template. It is a driver exception rendered to the caller.

    `/api/ask` with a sentinel DSN cannot connect to anything, so this exercises the failure path
    rather than the happy one, which is the point.
    """
    response = client.get("/api/ask", params={"q": "anything at all", "lang": "en"})
    for secret in FORBIDDEN:
        assert secret not in response.text, f"/api/ask served {secret!r} on a failure"


def test_the_template_context_carries_no_settings_object() -> None:
    """The structural half, kept because the behavioural half cannot cover a page not yet written.

    A new screen that passes `settings` would serve nothing today and would be one debug line from
    serving a DSN tomorrow, and no response-reading test can see a template nobody has added.
    """
    from parts_answer_gate.api.app import _context  # noqa: PLC0415 - private by intent

    settings = get_settings()
    context: dict[str, Any] = _context(settings)

    assert "settings" not in context
    flattened = repr(context)
    assert settings.database_url not in flattened
    assert not re.search(r"Settings\(", flattened), "a Settings object reached the template context"
