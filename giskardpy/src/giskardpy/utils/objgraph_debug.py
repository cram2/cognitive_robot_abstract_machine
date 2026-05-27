"""
Lightweight objgraph-based leak diagnostics for the giskard process.

Everything here is a no-op unless the environment variable ``GISKARD_OBJGRAPH``
is truthy ("1"/"true"/"yes"/"on"), so it is safe to leave the call sites in
place and costs nothing in normal runs.

Can be used to check memory leaks or if something derived from the world
(compiled functions, collision objects, copies of bodies/expressions, old
collision matrices, ...) accumulates per model update and is never freed,
slowing the control loop until motion degrades.

These helpers snapshot Python object growth around the relevant events so the
leaking type(s) become visible.

Usage
-----
Run the giskard script with::

    GISKARD_OBJGRAPH=1 ros2 run ... tracy_velocity.py
    # or: GISKARD_OBJGRAPH=1 python3 tracy_velocity.py

Optional tuning via env vars:
    GISKARD_OBJGRAPH_LIMIT     how many types to list (default 30)
    GISKARD_OBJGRAPH_PERIODIC  seconds between background growth dumps (0=off)
    GISKARD_OBJGRAPH_BACKREFS  comma-separated type names to dump backref chains
                               for (text, no graphviz needed) on each report
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import types
from typing import Iterable, List, Optional


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


_FORCE_ENABLE: bool = False

ENABLED: bool = _FORCE_ENABLE or _truthy(os.environ.get("GISKARD_OBJGRAPH"))
_LIMIT: int = int(os.environ.get("GISKARD_OBJGRAPH_LIMIT", "30"))
# _BACKREF_TYPES: List[str] = [
#     t.strip()
#     for t in os.environ.get("GISKARD_OBJGRAPH_BACKREFS", "").split(",")
#     if t.strip()
# ]
_BACKREF_TYPES = ["ForceZMonitor", "_ReachTopTask"]

_lock = threading.Lock()
_periodic_thread: Optional[threading.Thread] = None


def _objgraph():
    try:
        import objgraph  # noqa: WPS433 (lazy import so the dep is optional)

        return objgraph
    except ImportError:
        print(
            "[objgraph] GISKARD_OBJGRAPH is set but objgraph is not installed "
            "(pip install objgraph). Diagnostics disabled."
        )
        return None


def report_growth(label: str = "") -> None:
    """Print the types that reached a new peak live-count since the last call.

    Call this around the event you suspect of leaking (e.g. after each world
    model update). A type with a positive delta on every event is the leak.
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
        # show_growth() prints only types whose count hit a new max, with deltas.
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
    """Print live instance counts for every type whose name contains any of
    ``substrings``. Useful to watch specific suspects (e.g. "Compiled",
    "Collision", "Matrix", "World", "Body")."""
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


def _describe(obj) -> str:
    """Short, safe description of a referrer: its type plus a hint at what it is."""
    try:
        cls = type(obj)
        name = f"{cls.__module__}.{cls.__name__}"
    except Exception:
        return "?"
    try:
        if isinstance(obj, dict):
            keys = list(obj.keys())[:4]
            return f"dict(len={len(obj)}, keys~={keys})"
        if isinstance(obj, (list, tuple, set, frozenset)):
            return f"{cls.__name__}(len={len(obj)})"
    except Exception:
        pass
    return name


def _dump_backref_chain(
    objgraph, type_name: str, depth: int = None, max_frontier: int = 400
) -> None:
    """Show *what keeps the OLDEST (i.e. leaked) instance of ``type_name`` alive*.

    For each backward hop we print a histogram of referrer types — so the real
    external holder shows up even when an object is referenced by dozens of its
    own siblings (which would otherwise hide in a "+N more"). Uses a single
    ``gc.get_referrers(*frontier)`` scan per level (cheap, bounded by ``depth``),
    not objgraph.find_backref_chain (unbounded, hangs on big casadi graphs).

    Depth defaults to GISKARD_OBJGRAPH_BACKREF_DEPTH (or 4).
    """
    import collections

    if depth is None:
        depth = int(os.environ.get("GISKARD_OBJGRAPH_BACKREF_DEPTH", "6"))
    objs = objgraph.by_type(type_name)
    if not objs:
        print(f"[objgraph]   (no live instances of {type_name})")
        return
    # Oldest instance is the one that should already have been freed if there
    # were no leak, so its referrers point at the actual retainer.
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
        # Cap frontier so deeper levels stay cheap; drop bare container types from
        # the carry-forward set only if it would otherwise explode.
        frontier = referrers[:max_frontier]


def _typename(obj) -> str:
    try:
        cls = type(obj)
        return f"{cls.__module__}.{cls.__name__}"
    except Exception:
        return "?"


def start_periodic(interval_seconds: Optional[float] = None) -> None:
    """Optionally start a background thread that prints growth every N seconds.

    Off unless ``GISKARD_OBJGRAPH_PERIODIC`` (or the argument) is > 0. A periodic
    dump catches leaks even where there is no obvious per-event hook, but mixes
    in transient control-loop objects, so the per-event ``report_growth`` hook is
    usually clearer.
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
