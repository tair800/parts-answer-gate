"""Alembic environment.

Two things here are deliberate. The URL comes from `store.engine.database_url()` rather than from
`alembic.ini`, so there is exactly one place that decides which database this project talks to. And
`src` is placed on the path explicitly: the project is run from a virtual environment that does not
install it, and an `env.py` that only works when somebody remembered to set PYTHONPATH is a
migration step that fails in CI and nowhere else.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parts_answer_gate.store.engine import build_engine, database_url  # noqa: E402
from parts_answer_gate.store.schema import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_object(_object: object, name: str | None, type_: str, *_args: object) -> bool:
    """Keep autogenerate away from the HNSW index.

    Alembic renders a `postgresql_using='hnsw'` index correctly, but it cannot compare one: it sees
    a btree-shaped reflection and proposes to drop and recreate the vector index on every
    autogenerate. Rebuilding an HNSW graph as a side effect of an unrelated schema change is a
    several-minute surprise, so the index is owned by the initial migration and excluded from
    comparison. Its build parameters live in `store.schema.HNSW_BUILD_PARAMETERS`.
    """
    if type_ == "index" and name is not None:
        return not name.endswith("_hnsw")
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=_include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = build_engine(poolclass=pool.NullPool)
    with engine.connect() as connection:
        _run(connection)
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
