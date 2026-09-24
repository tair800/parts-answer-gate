#!/bin/sh
# Migrate, index, then serve.
#
# `set -e` deliberately absent. A seeding failure must not stop the container: the evidence screen is
# artifact-backed and works with no database at all, `/healthz` reports the real state, and a console
# that serves four screens and explains the fifth is worth more to a visitor than one that
# crash-loops. The failure is loud in the logs and visible at /healthz, which is where an operator
# looks.
#
# `exec` on the last line so uvicorn becomes PID 1 and receives SIGTERM directly. Without it the
# shell holds PID 1, swallows the signal, and every deploy waits out the platform's kill timeout.

echo "[entrypoint] preparing the index"
python /app/scripts/seed_index.py || echo "[entrypoint] indexing failed; serving anyway, /healthz will say so"

echo "[entrypoint] starting uvicorn on ${PORT:-8000}"
exec uvicorn parts_answer_gate.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
