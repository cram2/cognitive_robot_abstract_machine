"""objgraph-based memory leak diagnostics for the giskard process.

Every function is a no-op unless the environment variable ``GISKARD_OBJGRAPH``
is truthy ("1"/"true"/"yes")

Use it to find objects that accumulate on every world model update and are never
freed (compiled functions, collision objects, body and expression copies, old
collision matrices), which slows the control loop over time. The helpers
snapshot Python object growth around the suspect events so the leaking types
become visible.

Usage:
    GISKARD_OBJGRAPH=1 ros2 run ... tracy_velocity.py
    GISKARD_OBJGRAPH=1 python3 tracy_velocity.py

Environment variables:
    GISKARD_OBJGRAPH_LIMIT         how many types to list (default 30)
    GISKARD_OBJGRAPH_PERIODIC      seconds between background growth dumps (0 disables)
    GISKARD_OBJGRAPH_BACKREF_DEPTH how many referrer levels to walk (default 6)
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import types
from typing import Iterable, Optional


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


_FORCE_ENABLE: bool = False

ENABLED: bool = _FORCE_ENABLE or _truthy(os.environ.get("GISKARD_OBJGRAPH"))
_LIMIT: int = int(os.environ.get("GISKARD_OBJGRAPH_LIMIT", "30"))
# Types whose retainer chains are dumped on every growth report.
_BACKREF_TYPES = []

_lock = threading.Lock()
_periodic_thread: Optional[threading.Thread] = None


def _objgraph():
    try:
        import objgraph  # lazy import keeps objgraph an optional dependency

        return objgraph
    except ImportError:
        print(
            "[objgraph] GISKARD_OBJGRAPH is set but objgraph is not installed "
            "(pip install objgraph). Diagnostics disabled."
        )
        return None


def report_growth(label: str = "") -> None:
    """Print the types that reached a new peak live count since the last call.

    Call this around an event you suspect of leaking, for example after each
    world model update. A type with a positive delta on every event is the leak.
    """
    if not ENABLED:
        return
    objgraph = _objgraph()
    if objgraph is None:
        return
    with _lock:
        gc.collect()
        header = f" — {label}" if label else ""
        print(f"\n[objgraph] ===== growth since last report{header} =====", flush=True)
        # Prints only types whose live count reached a new maximum, with deltas.
        objgraph.show_growth(limit=_LIMIT)
        for type_name in _BACKREF_TYPES:
            _dump_backref_chain(objgraph, type_name)
        print("[objgraph] =============================================\n", flush=True)


def report_most_common(label: str = "") -> None:
    """Print the most common live object types (absolute counts, not deltas)."""
    if not ENABLED:
        return
    objgraph = _objgraph()
    if objgraph is None:
        return
    with _lock:
        gc.collect()
        header = f" — {label}" if label else ""
        print(f"\n[objgraph] ===== most common types{header} =====", flush=True)
        objgraph.show_most_common_types(limit=_LIMIT)
        print("[objgraph] =====================================\n", flush=True)


def count_types(substrings: Iterable[str], label: str = "") -> None:
    """Print live instance counts for every type whose name contains any of the
    given substrings. Use it to watch specific suspects such as "Compiled",
    "Collision", "Matrix", "World" or "Body"."""
    if not ENABLED:
        return
    objgraph = _objgraph()
    if objgraph is None:
        return
    subs = [s.lower() for s in substrings]
    with _lock:
        gc.collect()
        stats = objgraph.typestats()
        matched = {
            name: cnt
            for name, cnt in stats.items()
            if any(s in name.lower() for s in subs)
        }
        header = f" — {label}" if label else ""
        print(f"\n[objgraph] ===== watched type counts{header} =====", flush=True)
        for name, cnt in sorted(matched.items(), key=lambda kv: -kv[1]):
            print(f"[objgraph]   {cnt:7d}  {name}")
        print("[objgraph] =====================================\n", flush=True)


def _dump_backref_chain(
    objgraph, type_name: str, depth: int = None, max_frontier: int = 400
) -> None:
    """Show what keeps the oldest (and therefore leaked) instance of type_name alive.

    At each backward hop it prints a histogram of referrer types, so the real
    external holder is visible even when the object is referenced by many of its
    own siblings. It runs one gc.get_referrers() scan per level, bounded by
    depth, instead of objgraph.find_backref_chain, which is unbounded and hangs
    on large casadi graphs.

    Depth defaults to GISKARD_OBJGRAPH_BACKREF_DEPTH, or 6 if that is unset.
    """
    import collections

    if depth is None:
        depth = int(os.environ.get("GISKARD_OBJGRAPH_BACKREF_DEPTH", "6"))
    objs = objgraph.by_type(type_name)
    if not objs:
        print(f"[objgraph]   (no live instances of {type_name})")
        return
    # Without a leak the oldest instance would already be freed, so its
    # referrers point at whatever is actually retaining it.
    target = objs[0]
    print(
        f"[objgraph]   referrer-type histogram up from OLDEST {type_name} "
        f"({len(objs)} live):"
    )
    ignore_ids = {id(objs), id(objs[-1]), id(sys._getframe())}
    seen = {id(target)}
    frontier = [target]
    for level in range(depth):
        referrers = [
            r
            for r in gc.get_referrers(*frontier)
            if id(r) not in seen
            and id(r) not in ignore_ids
            and not isinstance(r, types.FrameType)
        ]
        for r in referrers:
            seen.add(id(r))
        if not referrers:
            break
        hist = collections.Counter(_typename(r) for r in referrers)
        indent = "    " * (level + 1)
        summary = ", ".join(f"{name}×{cnt}" for name, cnt in hist.most_common(12))
        print(f"[objgraph]   {indent}L{level + 1} ({len(referrers)}): {summary}")
        # Cap the frontier so deeper levels stay cheap.
        frontier = referrers[:max_frontier]


def _typename(obj) -> str:
    try:
        cls = type(obj)
        return f"{cls.__module__}.{cls.__name__}"
    except Exception:
        return "?"


def start_periodic(interval_seconds: Optional[float] = None) -> None:
    """Start a background thread that prints growth every interval_seconds.

    Does nothing unless GISKARD_OBJGRAPH_PERIODIC (or the argument) is greater
    than zero. A periodic dump catches leaks where there is no obvious per-event
    hook, but it mixes in transient control-loop objects, so the per-event
    report_growth hook is usually clearer.
    """
    global _periodic_thread
    if not ENABLED:
        return
    if interval_seconds is None:
        interval_seconds = float(os.environ.get("GISKARD_OBJGRAPH_PERIODIC", "0"))
    if interval_seconds <= 0 or _periodic_thread is not None:
        return

    def _loop():
        while True:
            time.sleep(interval_seconds)
            report_growth(label=f"periodic/{interval_seconds:g}s")

    _periodic_thread = threading.Thread(
        target=_loop, daemon=True, name="objgraph-periodic"
    )
    _periodic_thread.start()
    print(
        f"[objgraph] periodic growth reporting every {interval_seconds:g}s",
        flush=True,
    )
