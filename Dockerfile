FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.8
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python

ENV TRANSPORT=stdio \
    HOST=127.0.0.1 \
    PORT=8050 \
    MCP_OUTPUT_DIR=/app/output \
    UV_OFFLINE=1
EXPOSE 8050
CMD ["/app/.venv/bin/python", "/app/src/main.py"]