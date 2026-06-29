"""GC monitor tests: callback severity routing, idempotent install, snapshot."""

import gc
import logging

from app.observability import _gc_callback, _gc_start, gc_snapshot, install_gc_monitor


def _run_collection(gen: int, collected: int = 0, uncollectable: int = 0) -> None:
    info = {"generation": gen, "collected": collected, "uncollectable": uncollectable}
    _gc_callback("start", info)
    _gc_callback("stop", info)


def test_gen2_collection_logged_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="app.gc"):
        _run_collection(gen=2, collected=7)
    assert any("gen=2" in r.message and "collected=7" in r.message for r in caplog.records)


def test_uncollectable_logged_at_warning(caplog):
    with caplog.at_level(logging.INFO, logger="app.gc"):
        _run_collection(gen=2, uncollectable=3)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "uncollectable=3" in warnings[0].message


def test_gen0_not_logged_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="app.gc"):
        _run_collection(gen=0, collected=1)
    assert not caplog.records  # gen0/1 are DEBUG-only


def test_start_phase_records_timing():
    _gc_callback("start", {"generation": 1})
    assert 1 in _gc_start
    _gc_callback("stop", {"generation": 1})  # cleanup
    assert 1 not in _gc_start


def test_install_is_idempotent():
    install_gc_monitor()
    install_gc_monitor()
    assert gc.callbacks.count(_gc_callback) == 1
    gc.callbacks.remove(_gc_callback)  # leave global state clean for other tests


def test_snapshot_has_expected_keys():
    snap = gc_snapshot()
    assert set(snap) == {"count", "stats", "threshold"}
