"""Tests for IngestionQueue.flush() failure handling.

Bug: flush() used to clear self._pending *before* calling reindex_fn(batch)
with no try/except. If reindex_fn raised (e.g. a transient OpenAI/DB
error), the batch was already gone — silent document loss, no retry.

These tests prove that a reindex_fn failure requeues the batch instead
of losing it, and that the queue recovers cleanly on a subsequent
successful flush.
"""

from __future__ import annotations

import time

import pytest

from archrag.services.ingestion_queue import IngestionQueue


def test_flush_requeues_batch_when_reindex_fn_raises():
    """A failing reindex_fn must not lose the pending documents."""
    calls: list[list[dict]] = []

    def failing_reindex(batch: list[dict]) -> None:
        calls.append(batch)
        raise RuntimeError("simulated transient OpenAI/DB error")

    queue = IngestionQueue(reindex_fn=failing_reindex, flush_interval=999)
    queue.enqueue([{"text": "doc1"}, {"text": "doc2"}])

    with pytest.raises(RuntimeError, match="simulated transient"):
        queue.flush()

    # The batch must still be pending — not lost.
    assert queue.pending_count() == 2
    assert len(calls) == 1


def test_flush_succeeds_after_previous_failure_without_losing_or_duplicating():
    """After a failed flush, a later successful flush indexes the full batch exactly once."""
    attempts: list[list[dict]] = []

    def flaky_reindex(batch: list[dict]) -> None:
        attempts.append(list(batch))
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")
        # second attempt succeeds

    queue = IngestionQueue(reindex_fn=flaky_reindex, flush_interval=999)
    queue.enqueue([{"text": "doc1"}])

    with pytest.raises(RuntimeError):
        queue.flush()
    assert queue.pending_count() == 1

    flushed_count = queue.flush()
    assert flushed_count == 1
    assert queue.pending_count() == 0

    # reindex_fn was called twice, each time with exactly the one document
    # (no duplication, no loss).
    assert len(attempts) == 2
    assert attempts[0] == [{"text": "doc1"}]
    assert attempts[1] == [{"text": "doc1"}]


def test_flush_requeues_ahead_of_newly_enqueued_documents():
    """Docs enqueued *during* a failed flush must not be lost or reordered oddly:
    the failed batch goes back to the front of the queue."""

    def failing_reindex(batch: list[dict]) -> None:
        raise RuntimeError("boom")

    queue = IngestionQueue(reindex_fn=failing_reindex, flush_interval=999)
    queue.enqueue([{"text": "original"}])

    with pytest.raises(RuntimeError):
        queue.flush()

    # Simulate a new doc arriving after the failed flush.
    queue.enqueue([{"text": "new"}])

    assert queue.pending_count() == 2


def test_timer_loop_survives_flush_failure_and_retries_next_interval():
    """A transient reindex_fn failure during an *auto*-flush (triggered by
    the background timer thread, not a direct flush() call) must not kill
    the daemon thread. The next interval must retry and eventually
    succeed instead of auto-flush silently stopping forever."""
    attempts: list[list[dict]] = []

    def flaky_reindex(batch: list[dict]) -> None:
        attempts.append(list(batch))
        if len(attempts) == 1:
            raise RuntimeError("simulated transient failure during auto-flush")
        # second+ attempt succeeds

    queue = IngestionQueue(
        reindex_fn=flaky_reindex,
        flush_interval=0.05,
        poll_interval=0.02,
    )
    try:
        queue.enqueue([{"text": "doc1"}])

        # Wait for the first (failing) auto-flush attempt.
        deadline = time.monotonic() + 2.0
        while len(attempts) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(attempts) == 1, "first auto-flush attempt never happened"

        # The timer thread must still be alive after the failure.
        assert queue._timer.is_alive(), "timer thread died after auto-flush failure"

        # Wait for the second (successful) auto-flush attempt.
        deadline = time.monotonic() + 2.0
        while len(attempts) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(attempts) == 2, (
            "timer thread did not retry after the first auto-flush failure"
        )
        assert queue.pending_count() == 0
    finally:
        queue._shutdown = True
