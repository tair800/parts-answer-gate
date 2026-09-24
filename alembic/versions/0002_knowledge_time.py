"""knowledge time: the second bitemporal axis on document and chunk

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24

Validity alone answers *what was true of the machine in March*. It cannot answer *what did we know
in March*, and those are different questions with different right answers whenever a correction
lands after the period it describes. This migration adds the second axis.

`known_from` is backfilled from `valid_from` — every row already in the table was believed from the
day it came into force, which is true of every document that has never been corrected — and only
then made `NOT NULL`. Adding it as `NOT NULL` in one statement would fail against any non-empty
table, which is every deployment that has already loaded a corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("document", "chunk")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("known_from", sa.Date(), nullable=True))
        op.add_column(table, sa.Column("known_to", sa.Date(), nullable=True))
        op.add_column(table, sa.Column("corrected_by", sa.String(64), nullable=True))

        # Backfill before the NOT NULL, so this runs against a populated database.
        op.execute(f"UPDATE {table} SET known_from = valid_from WHERE known_from IS NULL")
        op.alter_column(table, "known_from", nullable=False)

    # Current knowledge — `known_to IS NULL` — is the common read, so it leads the chunk index.
    op.create_index("ix_document_known", "document", ["known_from", "known_to"])
    op.create_index("ix_chunk_known", "chunk", ["known_to", "known_from"])


def downgrade() -> None:
    op.drop_index("ix_chunk_known", table_name="chunk")
    op.drop_index("ix_document_known", table_name="document")
    for table in _TABLES:
        op.drop_column(table, "corrected_by")
        op.drop_column(table, "known_to")
        op.drop_column(table, "known_from")
