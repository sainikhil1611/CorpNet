"""ArchRAG FastMCP Server.

Exposes the full ArchRAG pipeline as MCP tools:
  - health, index, query, search, add, remove, reindex, info

The orchestrator and ingestion queue are initialised eagerly at
startup so every tool is ready to respond immediately.

The `add` tool enqueues documents into a thread-safe
IngestionQueue which auto-flushes every 3 minutes.
The `reindex` tool forces an immediate flush.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

# Walk up from this file (archrag/mcp_server.py) to the project root and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator
from archrag.services.ingestion_queue import IngestionQueue

log = logging.getLogger(__name__)

# ── Server singleton ──

mcp = FastMCP(
    "ArchRAG",
    instructions=(
        "ArchRAG: Attributed Community-based Hierarchical RAG server. "
        "Use 'index' to build from a corpus, 'query' to ask questions, "
        "'search' to find entities/chunks, 'add' to enqueue new documents "
        "(batched), and 'reindex' to flush the queue immediately."
    ),
)

# ── Eager initialisation (runs once at startup) ──

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_config_path = os.environ.get("ARCHRAG_CONFIG", "config.yaml")
log.info("Loading config from %s …", _config_path)

log.info("Building orchestrator (embedding, LLM, stores) …")
_orch = build_orchestrator(_config_path)
log.info("Orchestrator ready.")

log.info("Starting ingestion queue …")
_queue = IngestionQueue(
    reindex_fn=_orch.add_documents,
    flush_interval=float(os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180")),
)
log.info("Ingestion queue ready (flush every %s s).", os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180"))

log.info("All components initialised — MCP server is ready.")


# ── MCP Tools ──


@mcp.tool()
def index(corpus_path: str) -> str:
    """Build the full index from a corpus file (JSONL or JSON array).

    This wipes existing data and rebuilds from scratch.

    Args:
        corpus_path: Path to the corpus file on disk.
    """
    _orch.index(corpus_path)
    st = _orch.stats()
    return (
        f"Indexing complete. "
        f"{st['entities']} entities, {st['relations']} relations, "
        f"{st['chunks']} chunks, {st['hierarchy_levels']} hierarchy levels."
    )


@mcp.tool()
def query(question: str) -> str:
    """Answer a question using hierarchical search + adaptive filtering.

    Args:
        question: The natural-language question to answer.
    """
    return _orch.query(question)


@mcp.tool()
def search(
    query_str: str,
    search_type: str = "entities",
) -> str:
    """Search the knowledge graph by substring.

    Args:
        query_str: The search term (case-insensitive substring match).
        search_type: What to search — "entities", "chunks", or "all".
    """
    parts: list[str] = []

    if search_type in ("entities", "all"):
        entities = _orch.search_entities(query_str)
        if entities:
            lines = [f"  [{e['type']}] {e['name']}: {e['description'][:120]}" for e in entities]
            parts.append(f"Entities ({len(entities)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No entities matching '{query_str}'.")

    if search_type in ("chunks", "all"):
        chunks = _orch.search_chunks(query_str)
        if chunks:
            lines = [f"  {c['id']}: {c['text'][:120]}..." for c in chunks]
            parts.append(f"Chunks ({len(chunks)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No chunks matching '{query_str}'.")

    return "\n\n".join(parts)


@mcp.tool()
def add(documents: list[dict[str, Any]]) -> str:
    """Enqueue new documents for batched indexing.

    Documents are NOT indexed immediately — they sit in a pending
    queue and are flushed every 3 minutes, or when 'reindex' is called.

    Each document should have a "text" field (or "title"+"context").

    Args:
        documents: List of document dicts, each with at least a "text" key.
    """
    pending = _queue.enqueue(documents)
    return (
        f"Enqueued {len(documents)} document(s). "
        f"{pending} total pending. "
        f"They will be indexed on the next flush (every 3 min) or call 'reindex'."
    )


@mcp.tool()
def remove(entity_name: str) -> str:
    """Remove an entity (and its relations) from the knowledge graph.

    Args:
        entity_name: The exact name of the entity to delete.
    """
    if _orch.remove_entity(entity_name):
        return f"Removed entity '{entity_name}' and its relations."
    return f"Entity '{entity_name}' not found."


import threading

_reindex_lock = threading.Lock()
_reindex_status: str = "idle"  # "idle" | "running" | "done: ..."


def _bg_reindex_worker():
    """Runs entirely on a detached thread — no asyncio involvement."""
    global _reindex_status
    log.info("[reindex-bg] Worker thread started")
    try:
        log.info("[reindex-bg] Calling _queue.flush() ...")
        flushed = _queue.flush()
        log.info("[reindex-bg] flush() returned: %d doc(s) flushed", flushed)

        log.info("[reindex-bg] Calling _orch.stats() ...")
        st = _orch.stats()
        log.info(
            "[reindex-bg] Complete: %d doc(s). "
            "DB: %d entities, %d relations, %d chunks, %d levels.",
            flushed, st["entities"], st["relations"],
            st["chunks"], st["hierarchy_levels"],
        )
        _reindex_status = (
            f"done: {flushed} doc(s) reindexed. "
            f"{st['entities']} entities, {st['relations']} relations, "
            f"{st['chunks']} chunks, {st['hierarchy_levels']} levels."
        )
    except Exception as exc:
        log.exception("[reindex-bg] FAILED")
        _reindex_status = f"error: {exc}"


@mcp.tool()
def reindex() -> str:
    """Immediately flush the pending document queue and reindex.

    Runs in the background so other tools remain responsive.
    Use 'info' to check when it finishes (pending count drops to 0).
    """
    global _reindex_status

    log.info("[reindex] Tool called")

    count = _queue.pending_count()
    log.info("[reindex] Pending count: %d", count)
    if count == 0:
        last = _reindex_status
        log.info("[reindex] Queue empty. Last status: %s", last)
        return f"Queue is empty — nothing to reindex. Last reindex status: {last}"

    if _reindex_status == "running":
        log.info("[reindex] Already running, skipping")
        return "A reindex is already in progress. Use 'info' to check status."

    _reindex_status = "running"
    log.info("[reindex] Spawning background thread ...")
    t = threading.Thread(target=_bg_reindex_worker, name="reindex-bg", daemon=True)
    t.start()
    log.info("[reindex] Thread spawned (id=%s), returning immediately", t.ident)

    return (
        f"Reindex started in background for {count} document(s). "
        f"Use 'info' to check progress — pending count will drop to 0 when done."
    )


@mcp.tool()
def health() -> str:
    """Quick health check. Returns immediately without initialising the pipeline."""
    return "healthy"


@mcp.tool()
def info() -> str:
    """Show database statistics and queue status."""
    st = _orch.stats()
    pending = _queue.pending_count()
    return (
        f"Entities:         {st['entities']}\n"
        f"Relations:        {st['relations']}\n"
        f"Chunks:           {st['chunks']}\n"
        f"Hierarchy levels: {st['hierarchy_levels']}\n"
        f"Pending in queue: {pending}\n"
        f"Reindex status:   {_reindex_status}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
