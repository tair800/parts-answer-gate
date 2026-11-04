"""The DSN shapes managed providers actually hand out.

Every one of these was a real deployment failure rather than a hypothetical. `postgres://` is what
Render's `fromDatabase` emits and what Heroku taught everyone to expect; SQLAlchemy removed that
dialect in 1.4 and `create_engine` raises before a query runs. `postgresql://` is what Neon and
Supabase print in their connection dialogs; SQLAlchemy accepts it and resolves it to psycopg2,
which this project does not install.

Both fail at connect time with an error that names a driver rather than a URL, which is why this
test exists: the next person to paste a connection string should not have to rediscover it.
"""

from __future__ import annotations

import pytest
from sqlalchemy.pool import NullPool

from parts_answer_gate.store.engine import SESSION_SETTINGS, build_engine, normalise_database_url

PROVIDER_URLS = [
    # Render's fromDatabase / the Heroku convention.
    ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    # Neon, Supabase, and most "copy connection string" buttons.
    ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    # Managed providers require TLS, so the query string must survive untouched.
    (
        "postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/db?sslmode=require",
        "postgresql+psycopg://u:p@ep-x.eu-central-1.aws.neon.tech/db?sslmode=require",
    ),
    # Already explicit: left alone.
    ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
]


@pytest.mark.parametrize(("given", "expected"), PROVIDER_URLS)
def test_provider_urls_are_coerced_to_the_installed_driver(given: str, expected: str) -> None:
    assert normalise_database_url(given) == expected


def test_a_non_postgres_url_is_not_rewritten() -> None:
    """The function picks a driver; it does not decide what a URL means."""
    assert normalise_database_url("sqlite:///local.db") == "sqlite:///local.db"


@pytest.mark.parametrize(("given", "_expected"), PROVIDER_URLS)
def test_build_engine_accepts_every_provider_shape(given: str, _expected: str) -> None:
    """Constructing the engine must not raise. It does not connect, so no database is needed.

    This is the assertion that would have caught the failure: `create_engine` is where
    `NoSuchModuleError` is raised, long before anything is queried.
    """
    engine = build_engine(given)
    assert engine.dialect.driver == "psycopg"


def test_a_url_gets_the_same_engine_on_later_calls() -> None:
    """The API asks for its engine where each request opens its session. A new engine there is a
    new pool, so a connection returned by a request would never be handed to the next request."""
    url = "postgresql+psycopg://u:p@host/shared"
    assert build_engine(url) is build_engine(url)


def test_provider_shapes_of_the_same_database_share_its_engine() -> None:
    """The engine is keyed by the URL after normalisation, so the shape a provider hands out and the
    explicit driver reach the same pool."""
    assert build_engine("postgres://u:p@host/same") is build_engine(
        "postgresql+psycopg://u:p@host/same"
    )


def test_engine_options_get_an_engine_of_their_own() -> None:
    """A caller that passes `poolclass=NullPool` disposes what it gets back: that engine must never
    be the shared engine, nor be handed to a later caller."""
    url = "postgresql+psycopg://u:p@host/options"
    shared = build_engine(url)
    unpooled = build_engine(url, poolclass=NullPool)
    assert unpooled is not shared
    assert build_engine(url, poolclass=NullPool) is not unpooled
    assert build_engine(url) is shared


class _Connection:
    """A DBAPI connection reduced to the behaviour the session settings depend on.

    psycopg opens a transaction for the first statement on a connection that is not in autocommit
    mode and refuses to change autocommit inside one, and PostgreSQL undoes a `SET` made inside a
    transaction when that transaction rolls back.
    """

    def __init__(self) -> None:
        self._autocommit = False
        self.settings: dict[str, str] = {}
        self._before_transaction: dict[str, str] | None = None

    @property
    def autocommit(self) -> bool:
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        if self._before_transaction is not None:
            raise RuntimeError("cannot change autocommit inside a transaction")
        self._autocommit = value

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def execute(self, statement: str) -> None:
        if not self._autocommit and self._before_transaction is None:
            self._before_transaction = dict(self.settings)
        name, _, value = statement.removeprefix("SET ").partition(" = ")
        self.settings[name] = value

    def rollback(self) -> None:
        if self._before_transaction is not None:
            self.settings, self._before_transaction = self._before_transaction, None


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: str) -> None:
        self._connection.execute(statement)


def test_session_settings_survive_a_rollback_of_the_first_transaction() -> None:
    """A new connection's first transaction belongs to whoever uses it first. It is rolled back
    when `wait_for_database` hands the connection back, and `session_scope` rolls it back when its
    caller raises. Settings made inside that transaction went with it, and the pool kept a
    connection that searched at pgvector's defaults rather than at `SESSION_SETTINGS`."""
    engine = build_engine("postgresql+psycopg://host/session-settings")
    on_connect = [
        listener
        for listener in engine.pool.dispatch.connect
        if listener.__module__ == build_engine.__module__
    ]
    assert len(on_connect) == 1
    connection = _Connection()

    on_connect[0](connection, None)
    connection.rollback()

    assert connection.settings == SESSION_SETTINGS
    assert connection.autocommit is False
