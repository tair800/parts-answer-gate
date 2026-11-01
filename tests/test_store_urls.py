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

from parts_answer_gate.store.engine import build_engine, normalise_database_url

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
