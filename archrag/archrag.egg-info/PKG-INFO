Metadata-Version: 2.4
Name: archrag
Version: 0.1.0
Summary: ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation
License: MIT
Requires-Python: >=3.10
Description-Content-Type: text/markdown
Requires-Dist: click>=8.1
Requires-Dist: pyyaml>=6.0
Requires-Dist: numpy>=1.24
Requires-Dist: python-dotenv>=1.0
Requires-Dist: fastmcp>=2.0
Provides-Extra: ollama
Requires-Dist: ollama>=0.3; extra == "ollama"
Provides-Extra: openai
Requires-Dist: openai>=1.0; extra == "openai"
Provides-Extra: embeddings
Requires-Dist: sentence-transformers>=2.2; extra == "embeddings"
Provides-Extra: clustering
Requires-Dist: leidenalg>=0.10; extra == "clustering"
Requires-Dist: igraph>=0.11; extra == "clustering"
Requires-Dist: scikit-learn>=1.3; extra == "clustering"
Provides-Extra: all
Requires-Dist: ollama>=0.3; extra == "all"
Requires-Dist: openai>=1.0; extra == "all"
Requires-Dist: sentence-transformers>=2.2; extra == "all"
Requires-Dist: leidenalg>=0.10; extra == "all"
Requires-Dist: igraph>=0.11; extra == "all"
Requires-Dist: scikit-learn>=1.3; extra == "all"
Provides-Extra: dev
Requires-Dist: pytest>=7.0; extra == "dev"
Requires-Dist: pytest-cov>=4.0; extra == "dev"

# ArchRAG

An implementation of **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)).

Built with a **hexagonal / ports & adapters** architecture — every external dependency (LLM, embedding model, database, vector index, clustering algorithm) is behind an abstract port and can be swapped via config.

## Quick Start

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all,dev]"
```

### 2. Set your API key

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

### 3. Configure

Copy the example config and adjust as needed:

```powershell
copy config.example.yaml config.yaml
```

Default config uses OpenAI (`gpt-4o-mini` + `text-embedding-3-small`). See [config.example.yaml](config.example.yaml) for all adapter options (Ollama, SentenceTransformers, etc.).

### 4. Prepare a corpus

JSONL file, one document per line:

```jsonl
{"text": "Albert Einstein developed the theory of special relativity in 1905."}
{"text": "Marie Curie discovered polonium and radium."}
```

Also supports `{"title": "...", "context": "..."}` format from the original paper, and JSON arrays.

### 5. Run

```powershell
# Build the full index (KG → hierarchical clustering → C-HNSW)
archrag index corpus.jsonl

# Ask a question
archrag query "What did Einstein win the Nobel Prize for?"
```

## CLI Reference

| Command | Description |
|---|---|
| `archrag index <corpus>` | Build full index from a JSONL / JSON corpus file |
| `archrag query "<question>"` | Answer a question using hierarchical search + adaptive filtering |
| `archrag search "<term>"` | Search entities by name (substring match) |
| `archrag search "<term>" -t chunks` | Search raw text chunks |
| `archrag search "<term>" -t all` | Search both entities and chunks |
| `archrag add <corpus>` | Add new documents to an existing index and re-index |
| `archrag remove "<entity name>"` | Delete an entity and its relations from the KG |
| `archrag info` | Show database stats and current configuration |

Add `-v` for debug logging, `-c path/to/config.yaml` for a custom config:

```powershell
archrag -v -c my_config.yaml query "some question"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

```
CLI (click)
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

### Ports & Adapters

| Port | Adapters |
|---|---|
| **EmbeddingPort** | SentenceTransformers, OpenAI, Ollama |
| **LLMPort** | OpenAI, Ollama |
| **GraphStorePort** | SQLite, In-Memory |
| **DocumentStorePort** | SQLite, In-Memory |
| **VectorIndexPort** | NumPy (brute-force cosine) |
| **ClusteringPort** | Leiden (via igraph + leidenalg) |

## Project Structure

```
archrag/
├── domain/models.py          # Pure dataclasses (Entity, Relation, Community, etc.)
├── ports/                    # 6 abstract base classes
├── adapters/
│   ├── embeddings/           # SentenceTransformer, OpenAI, Ollama
│   ├── llms/                 # OpenAI, Ollama
│   ├── stores/               # SQLite & in-memory (graph + document)
│   ├── indexes/              # NumPy vector index
│   └── clustering/           # Leiden
├── services/                 # Business logic (KG, clustering, C-HNSW, search, filtering)
├── prompts/                  # LLM prompt templates
├── config.py                 # YAML config + adapter factory + dotenv loading
└── cli.py                    # Click CLI entry point
tests/                        # 21 unit tests with mock ports
```

## Tests

```powershell
python -m pytest tests/ -v
```

## Paper Reference

> **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation**
> [arXiv:2502.09891](https://arxiv.org/abs/2502.09891)
