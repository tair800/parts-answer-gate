# Two stages. The build stage has uv and a compiler; the runtime stage has neither, which is both
# smaller and a smaller attack surface — nothing in the shipped image can compile a wheel.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before source, so editing a Python file does not re-resolve the lock file.
# `--frozen` refuses to update it: an image whose dependencies drifted from uv.lock is not the thing
# that was tested.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# The embedding model, baked in at build time.
#
# ~220MB of ONNX weights. Downloading them on first request would make the first technician question
# after a cold start wait for a model host, and would make the service depend on that host being up
# for ever. Baking it in trades image size for a dependency the deployment does not have.
ENV PAG_EMBEDDING_CACHE=/app/.fastembed_cache
RUN /app/.venv/bin/python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', \
              cache_dir='/app/.fastembed_cache')"


FROM python:3.12-slim-bookworm AS runtime

# A non-root user with no home and no shell. The process reads questions from the public internet;
# it has no business being able to log in or write anywhere but /tmp.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin --uid 10001 parts

WORKDIR /app

COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --from=build --chown=root:root /app/src /app/src
COPY --from=build --chown=root:root /app/scripts /app/scripts
COPY --from=build --chown=root:root /app/.fastembed_cache /app/.fastembed_cache

# The migrations. Without these the container can bring up its schema only through a test helper,
# and the thing deployed would not be the thing that was tested.
COPY --chown=root:root alembic.ini ./alembic.ini
COPY --chown=root:root alembic ./alembic

# The evidence. `/evidence` reads these files and nothing else, so it is the one screen that works
# with no database at all — and it is the screen a reader should look at first. Leaving them out
# would deploy a console reporting "not measured" for every figure the README quotes.
COPY --chown=root:root artifacts ./artifacts

COPY --chown=root:root docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# `chmod` explicitly rather than trusting the mode git recorded. A shell script committed 100644
# from a Windows checkout lands non-executable on any platform that checks out from git, and the
# container exits 128 with no message naming the file. The git mode is 100755 as well, and a test
# asserts it; this line means the image is correct even when that is not.
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PAG_EMBEDDING_CACHE=/app/.fastembed_cache \
    # Read-only unless a deployment says otherwise, and no model key is baked in. An image shipping
    # one would put the same credential on every deployment that ever ran it.
    PAG_READ_ONLY=true

USER parts
EXPOSE 8000

# Answers 200 with a database and 503 without one. Both are answers; a container that hung would
# pass a build check and fail on first deploy. `PORT` is read from the environment because the
# platform assigns it — hard-coding 8000 would probe a port nothing is listening on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/healthz', timeout=4).status in (200, 503) else 1)"

CMD ["/usr/local/bin/docker-entrypoint.sh"]
