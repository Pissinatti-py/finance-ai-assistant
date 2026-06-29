"""Garbage-collector monitoring for application health.

This service keeps state in process memory (the conversation checkpointer and
the per-language graph cache), so memory pressure and leaks are a real concern.
A ``gc.callbacks`` hook logs collection events so they can be watched in the
logs / shipped to monitoring:

- generation-2 collections (the expensive, pause-inducing ones) at INFO, with
  their duration — useful to correlate latency spikes;
- uncollectable objects (reference cycles GC cannot free → likely leak) at
  WARNING;
- generation 0/1 collections at DEBUG (noisy; on only when LOG_LEVEL=DEBUG).
"""

import gc
import logging
import time

logger = logging.getLogger("app.gc")

# Per-generation collection start time, set on the "start" phase.
_gc_start: dict[int, float] = {}


def _gc_callback(phase: str, info: dict) -> None:
    """
    ``gc.callbacks`` hook: time collections and log them by severity.

    :param phase: ``"start"`` or ``"stop"``.
    :type phase: str
    :param info: GC info dict with ``generation``, ``collected``, ``uncollectable``.
    :type info: dict
    :return: Nothing.
    :rtype: None
    """
    gen = info.get("generation", -1)
    if phase == "start":
        _gc_start[gen] = time.perf_counter()
        return

    elapsed_ms = (time.perf_counter() - _gc_start.pop(gen, time.perf_counter())) * 1000
    collected = info.get("collected", 0)
    uncollectable = info.get("uncollectable", 0)
    counts = gc.get_count()

    if uncollectable:
        logger.warning(
            f"gc gen={gen} uncollectable={uncollectable} collected={collected} "
            f"elapsed={elapsed_ms:.1f}ms counts={counts}"
        )
    elif gen >= 2:
        logger.info(f"gc gen={gen} collected={collected} elapsed={elapsed_ms:.1f}ms counts={counts}")
    else:
        logger.debug(f"gc gen={gen} collected={collected} elapsed={elapsed_ms:.2f}ms counts={counts}")


def install_gc_monitor() -> None:
    """
    Register the GC logging callback (idempotent) and log current thresholds.

    :return: Nothing.
    :rtype: None
    """
    if _gc_callback not in gc.callbacks:
        gc.callbacks.append(_gc_callback)
        logger.info(f"gc monitor installed thresholds={gc.get_threshold()}")


def gc_snapshot() -> dict:
    """
    Return a point-in-time GC snapshot for on-demand logging or metrics.

    :return: ``{"count", "stats", "threshold"}`` from the ``gc`` module.
    :rtype: dict
    """
    return {"count": gc.get_count(), "stats": gc.get_stats(), "threshold": gc.get_threshold()}
