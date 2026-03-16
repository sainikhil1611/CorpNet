"""Thread-safe batched ingestion queue.

Documents added via `enqueue()` are held in a pending list and
flushed (indexed) either:
  - automatically every *flush_interval* seconds (default 180), or
  - immediately when `flush()` is called.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


class IngestionQueue:
    """Thread-safe document queue with periodic auto-flush.

    Parameters
    ----------
    reindex_fn:
        Callable that accepts ``list[dict]`` of documents and indexes
        them (e.g. ``orchestrator.add_documents``).
    flush_interval:
        Seconds between automatic flushes.  A background daemon thread
        checks every 10 s whether the interval has elapsed.
    """

    def __init__(
        self,
        reindex_fn: Callable[[list[dict[str, Any]]], None],
        flush_interval: float = 180.0,
    ) -> None:
        self._reindex_fn = reindex_fn
        self._flush_interval = flush_interval

        self._lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._shutdown = False

        # Daemon timer thread — dies when main process exits
        self._timer = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer.start()

    # ── public API ──

    def enqueue(self, documents: list[dict[str, Any]]) -> int:
        """Add documents to the pending queue.  Returns new pending count."""
        with self._lock:
            self._pending.extend(documents)
            count = len(self._pending)
        log.info("Enqueued %d doc(s); %d total pending", len(documents), count)
        return count

    def flush(self) -> int:
        """Immediately index all pending documents.  Returns count flushed."""
        with self._lock:
            if not self._pending:
                return 0
            batch = list(self._pending)
            self._pending.clear()
            self._last_flush = time.monotonic()

        log.info("Flushing %d document(s)", len(batch))
        self._reindex_fn(batch)
        return len(batch)

    def pending_count(self) -> int:
        """Number of documents waiting in the queue."""
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        """Flush remaining docs and stop the timer thread."""
        self._shutdown = True
        self.flush()
        log.info("IngestionQueue shut down")

    # ── background timer ──

    def _timer_loop(self) -> None:
        """Background loop: check every 10 s if flush_interval has elapsed."""
        while not self._shutdown:
            time.sleep(10)
            with self._lock:
                elapsed = time.monotonic() - self._last_flush
                has_pending = len(self._pending) > 0
            if has_pending and elapsed >= self._flush_interval:
                log.info("Auto-flush triggered (%.0f s elapsed)", elapsed)
                self.flush()
