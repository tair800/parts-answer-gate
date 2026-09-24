#!/bin/sh
# Migrate, index, then serve.
#
# The migration was missing. The header said "migrate, index, then serve", the Dockerfile copies
# `alembic/` and `alembic.ini` into the image specifically so this script can run it, and the script
# ran neither — it went straight to `seed_index.py`, which does not create a schema either. Against
# a fresh database the container came up with no tables at all. Migration 0002 makes it worse rather
# than better: an image that starts against a database at 0001 would fail every query touching
# knowledge time.
#
# `set -e` deliberately absent. A seeding failure must not stop the container: the evidence screen is
# artifact-backed and works with no database at all, `/healthz` reports the real state, and a console
# that serves four screens and explains the fifth is worth more to a visitor than one that
# crash-loops. The failure is loud in the logs and visible at /healthz, which is where an operator
# looks.
#
# `PAG_SKIP_SEED` skips both database steps. CI passes it to prove the image starts with no database
# reachable at all, and until now nothing read it, so that step was really proving that a 60-second
# connect timeout eventually expires.
#
# `exec` on the last line so uvicorn becomes PID 1 and receives SIGTERM directly. Without it the
# shell holds PID 1, swallows the signal, and every deploy waits out the platform's kill timeout.

if [ "${PAG_SKIP_SEED}" = "true" ]; then
  echo "[entrypoint] PAG_SKIP_SEED=true; skipping migration and indexing"
else
  echo "[entrypoint] bringing the schema up to head"
  (cd /app && python -m alembic upgrade head) || echo "[entrypoint] migration failed; serving anyway, /healthz will say so"

  echo "[entrypoint] preparing the index"
  python /app/scripts/seed_index.py --skip-if-populated || echo "[entrypoint] indexing failed; serving anyway, /healthz will say so"
fi

echo "[entrypoint] starting uvicorn on ${PORT:-8000}"
exec uvicorn parts_answer_gate.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
