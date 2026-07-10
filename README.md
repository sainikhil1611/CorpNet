# CorpNet

An implementation of **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)) — a graph-based RAG pipeline (Python, `archrag` package) built with a hexagonal / ports-&-adapters architecture. Every external dependency (LLM, embedding model, database, vector index, clustering algorithm) sits behind an abstract port and is swapped via `config.yaml`. Ships as a Click CLI (`archrag`) and a FastMCP server (`archrag/mcp_server.py`, stdio transport).

## Quick start (devcontainer — recommended)

1. Prereqs: [Docker](https://docs.docker.com/get-docker/), [VS Code](https://code.visualstudio.com/) + the "Dev Containers" extension.
2. Open this folder in VS Code → "Reopen in Container" when prompted. This installs the package in editable mode (`pip install --user -e ".[dev]"`).
3. `cp .env.example .env` and fill in `OPENAI_API_KEY` (see table below).
4. `cp config.example.yaml config.yaml` (or bring your own — the tracked `config.yaml` defaults to the OpenAI adapter). Then run:
   ```sh
   archrag index corpus.jsonl   # build the index from a sample corpus (create one — see "Manual setup")
   archrag query "What did Einstein win the Nobel Prize for?"
   ```
   This is a CLI/stdio tool, not a web server — there's no `localhost:<port>` to open.

## Manual setup (no devcontainer)

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Base install (click/pyyaml/numpy/python-dotenv/fastmcp only).
# Add the extras you need: [openai] [ollama] [embeddings] [clustering] or [all].
pip install -e ".[all,dev]"

cp .env.example .env             # fill in OPENAI_API_KEY if using the openai adapter
cp config.example.yaml config.yaml   # or edit the tracked config.yaml directly
```

Prepare a corpus (JSONL, one document per line):

```jsonl
{"text": "Albert Einstein developed the theory of special relativity in 1905."}
{"text": "Marie Curie discovered polonium and radium."}
```

Also supports `{"title": "...", "context": "..."}` (the format from the original paper) and JSON arrays.

Run the CLI:

```sh
archrag index corpus.jsonl                                   # build the full index
archrag query "What did Einstein win the Nobel Prize for?"   # ask a question
archrag search "Einstein"                                    # substring search entities
archrag add corpus.jsonl                                     # add docs to an existing index
archrag remove "Entity Name"                                  # delete an entity
archrag info                                                  # show DB stats + current config
```

Add `-v` for debug logging, `-c path/to/config.yaml` for a custom config.

Run the MCP server (exposes `health`, `index`, `query`, `search`, `add`, `remove`, `reindex`, `info` as MCP tools over stdio — requires `fastmcp`, already a base dependency):

```sh
python -m archrag.mcp_server
```

The server initialises the orchestrator eagerly at startup, so a missing `OPENAI_API_KEY` (or a config selecting an uninstalled adapter) now exits immediately with a clear message instead of a traceback. The `add` tool enqueues documents into a thread-safe `IngestionQueue` that auto-flushes every `ARCHRAG_FLUSH_INTERVAL` seconds (default 180) or on-demand via the `reindex` tool; a failed flush requeues the batch instead of losing it.

## Environment variables

| Variable | Required | Purpose | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | Only if `config.yaml` selects `adapter: openai` for `embedding` and/or `llm` (the tracked `config.yaml` does, by default) | Auth for the OpenAI embeddings/chat-completions endpoints | https://platform.openai.com/api-keys |
| `ARCHRAG_CONFIG` | No — defaults to `config.yaml` | Path to the config YAML, read by the MCP server (`archrag/mcp_server.py`) | n/a |
| `ARCHRAG_FLUSH_INTERVAL` | No — defaults to `180` | Seconds between automatic `IngestionQueue` flushes in the MCP server | n/a |

`config.example.yaml`'s default adapter is **Ollama** (no API key needed) but requires a reachable daemon at `http://localhost:11434` (`nomic-embed-text` + `llama3.1:8b`) — not bundled or verified here.

## Testing

```sh
python -m pytest -q
```

Expected: `35 passed` (24 pre-existing + 4 for the `IngestionQueue.flush()`/timer-loop requeue-and-retry-on-failure fix + 7 for the openai adapter/env validation fix). The suite is fully offline — every test uses in-memory/mocked ports (`MockLLM`, `MockEmbedding`, fake `openai`/`sentence_transformers` modules injected via `monkeypatch`), so no live OpenAI or Ollama calls are needed.

## Deployment

Build the production image: `docker build -t corpnet .`
Run it (mount your own `config.yaml`, corpus file, and a `data/` volume for persistence):
```sh
docker run --rm --env-file .env \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/corpus.jsonl:/app/corpus.jsonl:ro \
  -v $(pwd)/data:/app/data \
  corpnet index corpus.jsonl
```

Run the MCP server instead by overriding the entrypoint (the image's
`ENTRYPOINT` is `archrag`, so plain positional args get appended to it —
override `--entrypoint` too, not just the command, or Click rejects the
resulting `archrag python -m archrag.mcp_server`):
```sh
docker run --rm -i --env-file .env --entrypoint python3 corpnet -m archrag.mcp_server
```

This container runs on any host (Fly.io, Render, a VPS with Docker) — see
[DEPLOYMENT.md](../DEPLOYMENT.md) in the Projects root for the full walkthrough.

## Known limitations

- **OpenAI-dependent runtime paths are unverified without a real key.** The `openai` adapter's *clean-failure* behavior (missing package, missing `OPENAI_API_KEY`) is tested and verified end-to-end (unit tests + a live CLI/Docker run). Actually calling the OpenAI API (real `archrag index`/`query` with `adapter: openai`) was not exercised — no key was available in this environment.
- **Ollama default is unverified.** `config.example.yaml` requires a reachable Ollama daemon at `localhost:11434`; no daemon was available to test against.
- **The production image is large.** It installs the `[all]` extra (torch, sentence-transformers, scikit-learn, etc.) so any adapter combination works without a rebuild; trim to `[openai]`/`[ollama]` only if you know which adapters you'll use.
- **No HTTP port.** The CLI is invoked per-command; the MCP server speaks MCP over stdio, not HTTP — there's nothing to `curl` or publish a port for.

---

## Architecture

### Paper summary

**ArchRAG** is a graph-based RAG system with two phases:

**Offline indexing**
1. **KG Construction** — Chunk corpus → LLM extracts entities & relations → merge into a Knowledge Graph.
2. **LLM-based Hierarchical Clustering** — Augment KG (KNN edges by attribute similarity) → weighted community detection (Leiden) → LLM summarises each community → build higher-level graph of communities → repeat → produces a hierarchical tree Δ of Attributed Communities (ACs).
3. **C-HNSW Index** — Map entities (layer 0) and ACs (layers 1…L) to embeddings → build a Community-based HNSW index with intra-layer links (M nearest neighbours) and inter-layer links (nearest neighbour in adjacent layer).

**Online retrieval**
1. **Hierarchical Search** — Embed query → start from top layer, greedy traverse intra-layer links to find k nearest neighbours per layer, follow inter-layer links downward → collect results R₀…R_L.
2. **Adaptive Filtering-based Generation** — For each Rᵢ, LLM extracts an analysis report with relevance scores → sort and merge reports → LLM produces final answer.

### Hexagonal (ports & adapters) design

The core insight: anything that touches an external model or a persistent layer goes behind a port. The domain logic depends only on abstract interfaces; adapters are swapped via `config.yaml`.

```
CLI (click) / MCP server (fastmcp, stdio)
 │
 ▼
Orchestrator
 ├── KG Construction Service
 ├── Hierarchical Clustering Service  (Algorithm 1)
 ├── C-HNSW Build Service             (Algorithm 3)
 ├── Hierarchical Search Service       (Algorithm 2)
 └── Adaptive Filtering Service        (Equations 1 & 2)
      │
      ▼
   6 Ports (ABCs)
      │
      ▼
   Swappable Adapters
```

| Port | Responsibility | Default adapter | Swap-in options |
|------|---------------|------------------|-----------------|
| **EmbeddingPort** | Text → vector | `sentence_transformer` | `openai`, `ollama` |
| **LLMPort** | Prompt → completion | `ollama` | `openai` |
| **GraphStorePort** | Persist KG (entities + relations) | `sqlite` | `in_memory` (tests) |
| **DocumentStorePort** | Persist corpus chunks, community summaries, hierarchy metadata | `sqlite` | `in_memory` (tests) |
| **VectorIndexPort** | ANN index for C-HNSW | `numpy` | — |
| **ClusteringPort** | Weighted graph → communities | `leiden` | — |

### Key design decisions

1. **Pure domain** — `domain/models.py` has zero imports from adapters or external libs.
2. **Ports are ABCs** — every service constructor takes ports as arguments (dependency injection).
3. **Adapters are leaf nodes** — they import external libraries but nothing imports them except the config factory (`archrag/config.py`), and now via a clean `AdapterConfigError` if the openai adapter is selected without the package or `OPENAI_API_KEY`.
4. **Config-driven wiring** — `config.py` reads YAML → instantiates the right adapter for each port → passes them to services.
5. **C-HNSW in pure Python/NumPy** — avoids the custom FAISS fork from the paper; swappable to FAISS later via `VectorIndexPort`.
6. **Testability** — every service is tested with `InMemory*` adapters and mock LLM/embedding ports; the full suite runs offline in ~4s.

### Project structure

```
archrag/
├── domain/models.py          # Pure dataclasses (Entity, Relation, Community, etc.)
├── ports/                    # 6 abstract base classes
├── adapters/
│   ├── embeddings/           # SentenceTransformer, OpenAI, Ollama
│   ├── llms/                 # OpenAI, Ollama
│   ├── stores/                # SQLite & in-memory (graph + document)
│   ├── indexes/               # NumPy vector index
│   └── clustering/            # Leiden
├── services/                  # Business logic (KG, clustering, C-HNSW, search, filtering, ingestion queue)
├── prompts/                   # LLM prompt templates
├── config.py                  # YAML config + adapter factory + dotenv loading + clean adapter validation
├── cli.py                     # Click CLI entry point
└── mcp_server.py               # FastMCP server (stdio) — health/index/query/search/add/remove/reindex/info
tests/                          # pytest suite, all offline against mock ports
```

See [config.example.yaml](config.example.yaml) for the full annotated config template.

## Paper reference

> **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation**
> [arXiv:2502.09891](https://arxiv.org/abs/2502.09891)
