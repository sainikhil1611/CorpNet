# CorpNet

An implementation of **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)).

Built with a **hexagonal / ports & adapters** architecture — every external dependency (LLM, embedding model, database, vector index, clustering algorithm) is behind an abstract port and can be swapped via config.

## Quick Start

### 1. Create and activate the virtual environment

Create environment:
```powershell
python -m venv .venv
```

Use *one* of the three following commands to activate the environment, depending on your specific CLI.

   Activate environment (for Mac/Linux):
   ```sh
   source .venv/bin/activate
   ```

   Activate environment (for Windows Command Prompt):
   ```cmd
   .venv\Scripts\activate
   ```

   Activate environment (for Windows Powershell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

Install dependencies:
```powershell
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

### Paper Summary

**ArchRAG** (Attributed Community-based Hierarchical RAG) is a graph-based Retrieval-Augmented Generation system with two phases:

#### Offline Indexing
1. **KG Construction** — Chunk corpus → LLM extracts entities & relations → merge into a Knowledge Graph.
2. **LLM-based Hierarchical Clustering** — Augment KG (KNN edges by attribute similarity) → weighted community detection (Leiden) → LLM summarises each community → build higher-level graph of communities → repeat → produces a hierarchical tree Δ of Attributed Communities (ACs).
3. **C-HNSW Index** — Map entities (layer 0) and ACs (layers 1…L) to embeddings → build a Community-based HNSW index with intra-layer links (M nearest neighbours) and inter-layer links (nearest neighbour in adjacent layer).

#### Online Retrieval
1. **Hierarchical Search** — Embed query → start from top layer, greedy traverse intra-layer links to find k nearest neighbours per layer, follow inter-layer links downward → collect results R₀…R_L.
2. **Adaptive Filtering-based Generation** — For each Rᵢ, LLM extracts an analysis report with relevance scores → sort and merge reports → LLM produces final answer.

### Hexagonal (Ports & Adapters) Design

The core insight: **anything that touches an external model or a persistent layer goes behind a port**. The domain logic depends only on abstract interfaces. Adapters are swapped via configuration.

```
                    ┌─────────────────────────────────────┐
                    │          DOMAIN / SERVICES           │
                    │                                      │
                    │  KGConstructionService                │
                    │  HierarchicalClusteringService        │
                    │  CHNSWBuildService                    │
                    │  HierarchicalSearchService            │
                    │  AdaptiveFilteringService             │
                    │  ArchRAGOrchestrator                  │
                    │                                      │
                    │  Domain Models (Entity, Relation,     │
                    │   KnowledgeGraph, Community,          │
                    │   CommunityHierarchy, CHNSWIndex)     │
                    └──┬───┬───┬───┬───┬───┬───────────────┘
                       │   │   │   │   │   │
            ┌──────────┘   │   │   │   │   └──────────┐
            ▼              ▼   ▼   ▼   ▼              ▼
     ┌──────────┐  ┌──────┐ ┌─┐ ┌─┐ ┌──────┐  ┌──────────┐
     │EmbeddingP│  │LLM P │ │G│ │V│ │DocStr│  │Clustering│
     │   ort    │  │ ort  │ │r│ │e│ │Port  │  │   Port   │
     └────┬─────┘  └──┬───┘ │a│ │c│ └──┬───┘  └────┬─────┘
          │            │     │p│ │t│    │            │
          ▼            ▼     │h│ │o│    ▼            ▼
   ┌────────────┐ ┌────────┐│S│ │r│┌────────┐ ┌──────────┐
   │Nomic       │ │OpenAI  ││t│ │I││JSON    │ │Leiden    │
   │SentenceTfm │ │Ollama  ││o│ │n││SQLite  │ │Spectral  │
   │OpenAI Embed│ │Llama   ││r│ │d││        │ │SCAN      │
   └────────────┘ └────────┘│e│ │e│└────────┘ └──────────┘
                            │P│ │x│
                            │o│ │P│
                            │r│ │o│
                            │t│ │r│
                            └┬┘ │t│
                             │  └┬┘
                             ▼   ▼
                        ┌──────┐┌───────┐
                        │SQLite││Numpy  │
                        │Neo4j ││FAISS  │
                        └──────┘└───────┘
```

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

### Port Interfaces

| Port | Responsibility | Key Methods |
|------|---------------|-------------|
| **EmbeddingPort** | Text → vector | `embed(text) → list[float]`, `embed_batch(texts) → list[list[float]]` |
| **LLMPort** | Prompt → completion | `generate(prompt, system?) → str`, `generate_json(prompt, system?) → dict` |
| **GraphStorePort** | Persist KG (entities + relations) | `save_entities()`, `save_relations()`, `get_entity()`, `get_neighbours()`, `get_subgraph()` |
| **VectorIndexPort** | ANN index for C-HNSW | `add_vectors()`, `search()`, `save()`, `load()` |
| **DocumentStorePort** | Persist corpus chunks, community summaries, hierarchy metadata | `save_document()`, `get_document()`, `save_hierarchy()`, `load_hierarchy()` |
| **ClusteringPort** | Weighted graph → communities | `cluster(nodes, edges, weights) → list[set[str]]` |

### Ports & Adapters

| Port | Default Adapter | Swap-in Options |
|------|----------------|-----------------|
| **EmbeddingPort** | `SentenceTransformerAdapter` (nomic-embed-text) | `OpenAIEmbeddingAdapter`, `OllamaEmbeddingAdapter` |
| **LLMPort** | `OllamaAdapter` (llama3.1) | `OpenAIAdapter`, `AnthropicAdapter` |
| **GraphStorePort** | `SQLiteGraphStore` | `InMemoryGraphStore` (tests), future: Neo4j |
| **VectorIndexPort** | `NumpyVectorIndex` (pure-python C-HNSW) | `FAISSVectorIndex` |
| **DocumentStorePort** | `SQLiteDocumentStore` | `JSONDocumentStore`, `InMemoryDocStore` (tests) |
| **ClusteringPort** | `LeidenAdapter` | `SpectralClusteringAdapter`, `SCANAdapter` |

### Domain Models

| Model | Fields |
|-------|--------|
| **TextChunk** | id, text, metadata, source_doc |
| **Entity** | id, name, description, embedding? |
| **Relation** | id, source_id, target_id, description, weight? |
| **KnowledgeGraph** | entities: dict, relations: list |
| **Community** | id, level, member_entity_ids, summary, embedding? |
| **CommunityHierarchy** | levels: list[list[Community]], parent_map |
| **CHNSWLayer** | level, node_ids, intra_links, inter_links_down |
| **CHNSWIndex** | layers: list[CHNSWLayer], embeddings: dict |
| **SearchResult** | node_id, level, distance, text |
| **AnalysisReport** | points: list[{description, score}] |

### Services

| Service | Description |
|---------|-------------|
| **KGConstructionService** | Chunks corpus → LLM extracts entities/relations → persisted KG |
| **HierarchicalClusteringService** | Iterative: augment graph, cluster (Leiden), LLM summarises, repeat → CommunityHierarchy |
| **CHNSWBuildService** | Embeds entities + communities, builds intra/inter-layer links → CHNSWIndex |
| **HierarchicalSearchService** | Embeds query, traverses C-HNSW top-down → SearchResults per layer |
| **AdaptiveFilteringService** | LLM filters results per layer, merges, generates final answer |
| **ArchRAGOrchestrator** | Wires all services; blue/green snapshot for lock-free concurrent reads |

### Key Design Decisions

1. **Pure domain** — `models.py` has zero imports from adapters or external libs.
2. **Ports are ABCs** — every service constructor takes ports as arguments (dependency injection).
3. **Adapters are leaf nodes** — they import external libraries but nothing imports them except the config factory.
4. **Config-driven wiring** — `config.py` reads YAML → instantiates the right adapter for each port → passes them to services.
5. **C-HNSW in pure Python/NumPy** — avoids the custom FAISS fork from the paper; later swappable to FAISS via VectorIndexPort.
6. **Testability** — every service can be tested with `InMemory*` adapters and a mock LLM port.

## Configuration

```yaml
embedding:
  adapter: sentence_transformer  # | openai | ollama
  model: nomic-embed-text-v1.5
  dimension: 768

llm:
  adapter: ollama                # | openai
  model: llama3.1:8b
  base_url: http://localhost:11434
  temperature: 0.0

graph_store:
  adapter: sqlite                # | in_memory
  path: data/archrag.db

document_store:
  adapter: sqlite                # | in_memory | json
  path: data/archrag.db

vector_index:
  adapter: numpy                 # | faiss
  distance_metric: cosine

clustering:
  adapter: leiden                # | spectral | scan
  resolution: 1.0

indexing:
  chunk_size: 1200
  chunk_overlap: 100
  max_hierarchy_levels: 5
  knn_k: auto                   # auto = avg node degree
  similarity_threshold: 0.7

retrieval:
  k_per_layer: 5
  ef_search: 100

chnsw:
  M: 32                         # max connections per node
  ef_construction: 100
```

See [config.example.yaml](config.example.yaml) for the full annotated template.

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
