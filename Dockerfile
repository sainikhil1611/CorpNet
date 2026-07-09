FROM python:3.12-slim-bookworm
WORKDIR /app

# Install dependencies + package. `[all]` pulls every adapter extra
# (openai, ollama, sentence-transformers, leidenalg/igraph/scikit-learn)
# so any adapter selected in config.yaml works without a rebuild.
COPY pyproject.toml README.md ./
COPY archrag/ ./archrag/
RUN pip install --no-cache-dir ".[all]"

# Ship the safe Ollama-based example config only — operators must supply
# their own config.yaml (mount it, or set ARCHRAG_CONFIG) rather than
# bake a paid-API adapter selection into the image.
COPY config.example.yaml ./

# No network port: the CLI is invoked per-command, and the MCP server
# (python -m archrag.mcp_server) speaks MCP over stdio, not HTTP.
ENTRYPOINT ["archrag"]
CMD ["--help"]
