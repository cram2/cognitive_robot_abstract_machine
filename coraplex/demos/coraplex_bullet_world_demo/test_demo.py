#!/usr/bin/env python
import json
import time
import traceback

from segmind.event_segmentation import Segmind

PHASES = {"construct": 0.0, "first_tick": None, "tick_wall": 0.0, "tick_cpu": 0.0, "ticks": 0, "stop": 0.0}


def _timed(name, key):
    original = getattr(Segmind, name)

    def wrapper(self, *args, **kwargs):
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - wall
            if key == "tick_wall":
                PHASES["ticks"] += 1
                PHASES["tick_cpu"] += time.thread_time() - cpu
                if PHASES["first_tick"] is None:
                    PHASES["first_tick"] = elapsed
            PHASES[key] += elapsed

    setattr(Segmind, name, wrapper)


def main() -> None:
    """
    Run the bullet world demo and exit non-zero with a traceback on failure.
    """
    _timed("__post_init__", "construct")
    _timed("tick", "tick_wall")
    _timed("stop", "stop")
    started = time.monotonic()
    try:
        import demo

        demo.main()
    except Exception:
        traceback.print_exc()
        exit(1)
    finally:
        PHASES["demo"] = time.monotonic() - started
        rounded = {k: round(v, 2) if isinstance(v, float) else v for k, v in PHASES.items()}
        print(f"::notice title=segmind phases::{json.dumps(rounded)}", flush=True)


if __name__ == "__main__":
    main()
