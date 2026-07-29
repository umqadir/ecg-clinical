"""Exclusive lock for a single shared accelerator.

Any heavy training or inference run acquires this lock before touching the device
and releases it on exit, so concurrent runs on one machine serialize instead of
contending. Acquisition blocks and polls when the lock is held, and a lock whose
owner marker is older than ``stale_after_seconds`` is treated as abandoned and
cleared. The lock affects scheduling only and never a computed value; the
registered protocol fixes its path at ``/tmp/ml-train.lock`` and every entry point
accepts an override.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_device_lock(
    path: Path, *, poll_seconds: float = 30.0, stale_after_seconds: float = 12 * 3600.0
):
    marker = path / "owner.json"
    while True:
        try:
            path.mkdir()
            marker.write_text(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}) + "\n")
            break
        except FileExistsError:
            age: float | None = None
            try:
                payload = json.loads(marker.read_text())
                age = time.time() - float(payload["acquired_at"])
            except (FileNotFoundError, ValueError, KeyError):
                age = None
            if age is not None and age > stale_after_seconds:
                logging.warning("clearing stale device lock (age %.0fs): %s", age, path)
                marker.unlink(missing_ok=True)
                path.rmdir()
                continue
            logging.info("device lock held; waiting %.0fs: %s", poll_seconds, path)
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        # Remove the owner marker first, then the now-empty directory. A recursive
        # delete is silently refused under some restricted filesystem policies and
        # would leak the lock while appearing to succeed.
        marker.unlink(missing_ok=True)
        path.rmdir()
