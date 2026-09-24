"""Connecting to PostgreSQL, and the two session settings that decide whether HNSW is usable.

The interesting part of this module is `_configure_session`. pgvector's HNSW index is only consulted
for an `ORDER BY ... LIMIT` query, and by default an index scan stops after `ef_search` candidates —
which, with a `WHERE` clause that discards most of them, can return fewer rows than were asked for
or push the planner onto a sequential scan instead. `hnsw.iterative_scan` is pgvector 0.8's answer:
the index scan resumes until the limit is satisfied *after* filtering. Setting it on every
connection is what makes "effectivity filter in SQL, then an indexed vector search over what
survives" a real plan rather than a description.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

__all__ = [
    "DATABASE_URL_ENV",
    "DEFAULT_DATABASE_URL",
    "SESSION_SETTINGS",
    "build_engine",
    "database_url",
    "normalise_database_url",
    "session_scope",
    "wait_for_database",
]

DATABASE_URL_ENV: Final = "PAG_DATABASE_URL"

#: Matches `docker-compose.yml`. Port 15440 belongs to this project alone in this workspace.
DEFAULT_DATABASE_URL: Final = (
    "postgresql+psycopg://pag:pag_local_dev@127.0.0.1:15440/parts_answer_gate"
)

#: Applied to every connection, and published because they change measured recall.
#:
#: `hnsw.ef_search = 100` against pgvector's default of 40: the search beam. Wider costs latency and
#: buys recall, and kill condition F is a recall floor with a p95 that is nowhere near binding at
#: this corpus size.
#:
#: `hnsw.iterative_scan = strict_order` rather than `relaxed_order`: relaxed returns rows slightly
#: out of distance order, which would make two runs of the same query capable of producing different
#: rankings. Kill condition J requires byte-identical retrieval across runs, so the ordering
#: guarantee is worth more here than the throughput.
SESSION_SETTINGS: Final[dict[str, str]] = {
    "hnsw.ef_search": "100",
    "hnsw.iterative_scan": "strict_order",
}


def normalise_database_url(url: str) -> str:
    """Coerce a provider's DSN into the driver this project actually installs.

    Every managed PostgreSQL hands out a URL this codebase cannot use as given, and each does it
    differently:

    - **`postgres://`** — Render's `fromDatabase` and Heroku's convention. SQLAlchemy removed that
      dialect name in 1.4, so `create_engine` raises `NoSuchModuleError` before a single query runs.
    - **`postgresql://`** — Neon, Supabase, and most connection-string dialogs. SQLAlchemy accepts
      it and resolves it to **psycopg2**, which this project does not install: it pins psycopg 3.
      The failure is `ModuleNotFoundError: No module named 'psycopg2'` at connect time, which reads
      like a missing dependency rather than a URL that needs one more word in it.

    Only the scheme is touched. Everything after it — user, host, database, and the query string
    that carries `sslmode=require`, which every managed provider needs — is passed through
    unchanged, because rewriting any of that would be this function deciding how to connect rather
    than which driver connects.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def database_url() -> str:
    return normalise_database_url(os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL))


def build_engine(url: str | None = None, *, echo: bool = False, **kwargs: Any) -> Engine:
    engine = create_engine(
        normalise_database_url(url) if url else database_url(),
        echo=echo,
        # Retrieval opens a session per request and holds it for milliseconds; a pool that recycles
        # rather than reconnects is the difference between 1ms and 30ms of connect time per query,
        # which would dominate the published p95 and measure the pool instead of the index.
        pool_pre_ping=True,
        future=True,
        **kwargs,
    )
    _configure_session(engine)
    return engine


def _configure_session(engine: Engine) -> None:
    """Apply `SESSION_SETTINGS` on every new connection.

    On connect rather than per query: a `SET LOCAL` inside the retrieval path would have to be
    repeated by every caller, including the diagnostic that runs `EXPLAIN`, and a plan explained
    under different settings than the query executes under is not evidence of anything.
    """

    @event.listens_for(engine, "connect")
    def _set_guc(dbapi_connection: Any, _record: Any) -> None:
        with dbapi_connection.cursor() as cursor:
            for name, value in SESSION_SETTINGS.items():
                # Identifiers cannot be bound parameters, and these come from a module constant, not
                # from a caller.
                cursor.execute(f"SET {name} = {value}")


def wait_for_database(engine: Engine, *, timeout_seconds: float = 60.0) -> float:
    """Block until the database answers, returning how long that took.

    A freshly started container accepts TCP before it accepts queries. Returning the elapsed time
    rather than logging it lets a caller record a real number in an artifact.
    """
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return time.monotonic() - started
        except OperationalError as error:
            last = error
            time.sleep(0.5)
    raise TimeoutError(
        f"database at {engine.url.render_as_string(hide_password=True)} did not answer within "
        f"{timeout_seconds}s: {last}"
    )


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A transactional session. Commits on success, rolls back on any exception."""
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
