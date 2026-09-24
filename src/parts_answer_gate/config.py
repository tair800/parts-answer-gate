"""Configuration, read once, with no silent default for anything that costs money or writes.

Two settings carry weight and both fail closed.

`read_only` defaults to **true**, so a deployment that configured nothing serves the console and
accepts no ingestion. The cost of an over-restrictive default is a bug report; the cost of the other
kind is an anonymous visitor rebuilding an index.

`llm_api_key` has **no default at all**, and the abstractive answerer raises rather than degrading
when it is absent. That is deliberate: an arm that quietly returned nothing without a key would look
identical to an arm that ran and found nothing, and the evaluation would report a topology it never
called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from parts_answer_gate.store.engine import DATABASE_URL_ENV, DEFAULT_DATABASE_URL

__all__ = ["Settings", "get_settings"]

# `DEFAULT_DATABASE_URL` is imported rather than declared here, and the import direction is
# deliberate: `store/engine.py` is what actually opens a connection, and a settings module holding a
# second copy of the string would be a second thing to keep in step with docker-compose.yml. This
# file had its own copy with different credentials for about an hour, and nothing would have
# connected out of the box.


def _flag(name: str, *, default: bool) -> bool:
    """A boolean from the environment, strictly.

    Anything unrecognised is an error rather than a fallback. `PAG_READ_ONLY=flase` reading as true
    would be harmless; the same typo on a setting that defaulted the other way would not, and a
    parser that guesses is the wrong habit to have in one place and not the other.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean; use true or false")


@dataclass(frozen=True)
class Settings:
    database_url: str
    read_only: bool
    llm_api_key: str | None
    corpus_dir: str
    artifacts_dir: str
    embedding_cache_dir: str
    environment: str

    @property
    def live_model_available(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL),
        read_only=_flag("PAG_READ_ONLY", default=True),
        llm_api_key=os.environ.get("PAG_LLM_API_KEY") or None,
        corpus_dir=os.environ.get("PAG_CORPUS_DIR", "data/generated"),
        artifacts_dir=os.environ.get("PAG_ARTIFACTS_DIR", "artifacts"),
        embedding_cache_dir=os.environ.get("PAG_EMBEDDING_CACHE", ".fastembed_cache"),
        environment=os.environ.get("PAG_ENVIRONMENT", "local"),
    )
