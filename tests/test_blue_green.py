"""Blue/green snapshot concurrency integration test.

This test PROVES that readers see consistent data while a writer is
mid-reindex by:

1. Building an initial index with known data ("v1").
2. Starting a slow reindex on a background thread that injects a
   real `time.sleep()` between the clone and the swap, simulating
   the OpenAI calls that happen during a real reindex.
3. Firing off reader threads during that sleep window.
4. Asserting that every reader got v1 data — not partial, not empty,
   not crashed.
5. After the writer finishes, asserting new readers see v2 data.

Run with:
    python -m pytest tests/test_blue_green.py -v -s
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from archrag.adapters.indexes.numpy_vector import NumpyVectorIndex
from archrag.adapters.stores.in_memory_document import InMemoryDocumentStore
from archrag.adapters.stores.in_memory_graph import InMemoryGraphStore
from archrag.services.orchestrator import ArchRAGOrchestrator
from tests.conftest import MockClustering, MockEmbedding, MockLLM


# ── helpers ──


def _make_corpus_file(docs: list[dict]) -> str:
    """Write docs to a temp JSONL file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for doc in docs:
        f.write(json.dumps(doc) + "\n")
    f.close()
    return f.name


def _build_orchestrator() -> ArchRAGOrchestrator:
    return ArchRAGOrchestrator(
        llm=MockLLM(),
        embedding=MockEmbedding(),
        graph_store=InMemoryGraphStore(),
        doc_store=InMemoryDocumentStore(),
        vector_index=NumpyVectorIndex(),
        clustering=MockClustering(),
        chunk_size=500,
        chunk_overlap=50,
        max_levels=2,
        M=4,
        ef_construction=10,
        k_per_layer=3,
    )


# ── The actual tests ──


class TestBlueGreenSwap:
    """Prove that readers are never blocked or corrupted during reindex."""

    def test_readers_see_v1_while_writer_builds_v2(self):
        """
        Timeline:
          t=0.0  index v1 (fast, mocked)
          t=0.1  start reindex v2 on background thread
                 (we inject a 2s sleep inside _build_snapshot)
          t=0.2  fire 10 reader threads doing search_entities + stats
          t=0.3  readers all return v1 data ✓
          t=2.1  writer finishes, swap happens
          t=2.2  new reader sees v2 data ✓
        """
        orch = _build_orchestrator()

        # ── v1: index initial corpus ──
        v1_corpus = _make_corpus_file([
            {"title": "V1", "context": "Alice works at MIT with Bob.", "id": 0},
        ])
        try:
            orch.index(v1_corpus)
        finally:
            Path(v1_corpus).unlink(missing_ok=True)

        v1_stats = orch.stats()
        v1_entity_count = v1_stats["entities"]
        v1_chunk_count = v1_stats["chunks"]

        assert v1_entity_count > 0, "v1 should have entities"
        assert v1_chunk_count > 0, "v1 should have chunks"

        print(f"\n[v1] entities={v1_entity_count}, chunks={v1_chunk_count}")

        # ── inject a delay into the writer path ──
        # We monkey-patch _build_snapshot so the SECOND call (inside
        # index()) sleeps for 2 seconds, simulating slow OpenAI calls.
        original_build = orch._build_snapshot
        call_count = 0
        delay_seconds = 2.0

        def slow_build(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            snap = original_build(*args, **kwargs)
            # The writer calls _build_snapshot twice inside index():
            #   1st: to create the shadow with empty chnsw_index
            #   2nd: to create final snapshot with chnsw_index
            # We sleep on BOTH to simulate real work between clone and swap.
            if call_count >= 2:  # skip the __init__ call
                print(f"  [writer] _build_snapshot call #{call_count}, sleeping {delay_seconds}s...")
                time.sleep(delay_seconds)
            return snap

        # ── v2: start reindex on a background thread ──
        v2_corpus = _make_corpus_file([
            {"title": "V2", "context": "Charlie studies physics at Stanford.", "id": 0},
            {"title": "V2b", "context": "Diana leads research at Google DeepMind.", "id": 1},
        ])

        writer_error: Exception | None = None
        writer_done = threading.Event()

        def writer():
            nonlocal writer_error
            try:
                orch._build_snapshot = slow_build
                orch.index(v2_corpus)
            except Exception as exc:
                writer_error = exc
            finally:
                writer_done.set()

        writer_thread = threading.Thread(target=writer, name="writer-v2")
        writer_thread.start()

        # Give the writer a moment to enter the write lock and start building
        time.sleep(0.3)

        # ── fire reader threads while writer is sleeping ──
        reader_results: list[dict] = []
        reader_errors: list[Exception] = []
        reader_lock = threading.Lock()

        def reader(reader_id: int):
            try:
                stats = orch.stats()
                entities = orch.search_entities("Alice")
                result = {
                    "reader_id": reader_id,
                    "entity_count": stats["entities"],
                    "chunk_count": stats["chunks"],
                    "alice_found": len(entities) > 0,
                    "timestamp": time.monotonic(),
                }
                with reader_lock:
                    reader_results.append(result)
            except Exception as exc:
                with reader_lock:
                    reader_errors.append(exc)

        reader_threads = []
        for i in range(10):
            t = threading.Thread(target=reader, args=(i,), name=f"reader-{i}")
            reader_threads.append(t)
            t.start()

        for t in reader_threads:
            t.join(timeout=5)

        # ── assert: all readers got v1 data, no errors ──
        assert len(reader_errors) == 0, f"Reader errors: {reader_errors}"
        assert len(reader_results) == 10, f"Expected 10 results, got {len(reader_results)}"

        for r in reader_results:
            print(f"  [reader-{r['reader_id']}] entities={r['entity_count']}, "
                  f"chunks={r['chunk_count']}, alice_found={r['alice_found']}")
            # Every reader should see v1 counts — NOT 0, NOT partial
            assert r["entity_count"] == v1_entity_count, (
                f"Reader {r['reader_id']} saw {r['entity_count']} entities, "
                f"expected {v1_entity_count} (v1)"
            )
            assert r["chunk_count"] == v1_chunk_count
            assert r["alice_found"] is True, "Alice should exist in v1"

        print("  ✓ All 10 readers saw consistent v1 data during reindex")

        # ── wait for writer to finish ──
        writer_done.wait(timeout=10)
        writer_thread.join(timeout=5)
        Path(v2_corpus).unlink(missing_ok=True)

        assert writer_error is None, f"Writer crashed: {writer_error}"

        # ── assert: post-swap readers see v2 data ──
        v2_stats = orch.stats()
        print(f"  [v2] entities={v2_stats['entities']}, chunks={v2_stats['chunks']}")

        # MockLLM always extracts the same 3 entity names (Alice, Bob, MIT),
        # so entity count won't change.  But the CHUNKS are real — v2 corpus
        # has 2 docs, so chunk count must be 2 (v1 had 1).
        assert v2_stats["chunks"] == 2, (
            f"v2 should have 2 chunks (2 docs), got {v2_stats['chunks']}"
        )
        assert v2_stats["chunks"] != v1_chunk_count, (
            "v2 chunk count should differ from v1"
        )

        # The v2 corpus text mentions "Charlie" and "Stanford" — search
        # chunks to confirm the new text is there.
        charlie_chunks = orch.search_chunks("Charlie")
        assert len(charlie_chunks) > 0, "v2 chunk text 'Charlie' should be searchable"

        print("  ✓ Post-swap readers see v2 data (chunks updated, new text searchable)")

    def test_concurrent_readers_never_crash(self):
        """Hammer the orchestrator with 50 concurrent readers during a write.

        This is a stress test — we don't check exact values, just that
        nobody crashes, nobody gets an empty/None snapshot.
        """
        orch = _build_orchestrator()

        corpus = _make_corpus_file([
            {"title": "Stress", "context": "Einstein developed special relativity.", "id": 0},
        ])
        try:
            orch.index(corpus)
        finally:
            Path(corpus).unlink(missing_ok=True)

        errors: list[Exception] = []
        error_lock = threading.Lock()
        barrier = threading.Barrier(51)  # 50 readers + 1 writer

        def reader():
            barrier.wait(timeout=5)  # start all at once
            for _ in range(20):  # each reader does 20 rapid reads
                try:
                    stats = orch.stats()
                    assert stats["entities"] > 0, "Got 0 entities!"
                    assert stats["chunks"] > 0, "Got 0 chunks!"
                    orch.search_entities("Einstein")
                except Exception as exc:
                    with error_lock:
                        errors.append(exc)

        def writer():
            barrier.wait(timeout=5)
            new_corpus = _make_corpus_file([
                {"title": "New", "context": "Feynman worked on QED.", "id": 0},
                {"title": "New2", "context": "Bohr proposed the atomic model.", "id": 1},
            ])
            try:
                orch.add_documents([
                    {"title": "Added", "context": "Curie discovered radium.", "id": 99},
                ])
            finally:
                Path(new_corpus).unlink(missing_ok=True)

        threads = [threading.Thread(target=reader, name=f"stress-r{i}") for i in range(50)]
        threads.append(threading.Thread(target=writer, name="stress-w"))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"{len(errors)} reader(s) crashed: {errors[:3]}"
        print(f"\n  ✓ 50 readers × 20 reads + 1 writer = {50*20} reads, 0 crashes")

    def test_writer_serialisation(self):
        """Two concurrent writers should not interleave — _write_lock serialises them."""
        orch = _build_orchestrator()

        corpus = _make_corpus_file([
            {"title": "Init", "context": "Starting data.", "id": 0},
        ])
        try:
            orch.index(corpus)
        finally:
            Path(corpus).unlink(missing_ok=True)

        write_order: list[str] = []
        order_lock = threading.Lock()

        original_build = orch._build_snapshot

        def tracked_build(writer_name):
            def wrapper(*args, **kwargs):
                with order_lock:
                    write_order.append(f"{writer_name}-enter")
                snap = original_build(*args, **kwargs)
                time.sleep(0.5)  # hold the lock
                with order_lock:
                    write_order.append(f"{writer_name}-exit")
                return snap
            return wrapper

        def writer_a():
            orch._build_snapshot = tracked_build("A")
            orch.add_documents([{"title": "A", "context": "Writer A data.", "id": 10}])

        def writer_b():
            time.sleep(0.1)  # let A start first
            orch._build_snapshot = tracked_build("B")
            orch.add_documents([{"title": "B", "context": "Writer B data.", "id": 11}])

        ta = threading.Thread(target=writer_a, name="writer-A")
        tb = threading.Thread(target=writer_b, name="writer-B")
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)

        # The enters/exits should not interleave:
        # Valid: A-enter, A-exit, ..., B-enter, B-exit, ...
        # Invalid: A-enter, B-enter, ...
        print(f"\n  Write order: {write_order}")

        # Find first A-enter and first B-enter
        a_enters = [i for i, x in enumerate(write_order) if x == "A-enter"]
        a_exits = [i for i, x in enumerate(write_order) if x == "A-exit"]
        b_enters = [i for i, x in enumerate(write_order) if x == "B-enter"]

        if a_enters and a_exits and b_enters:
            # All of A's exits should come before B's first enter
            # (i.e. A fully completes before B starts)
            last_a_exit = max(a_exits)
            first_b_enter = min(b_enters)
            assert last_a_exit < first_b_enter, (
                f"Writers interleaved! A's last exit at index {last_a_exit}, "
                f"B's first enter at {first_b_enter}.\nFull order: {write_order}"
            )
            print("  ✓ Writers serialised correctly (A completed before B started)")
