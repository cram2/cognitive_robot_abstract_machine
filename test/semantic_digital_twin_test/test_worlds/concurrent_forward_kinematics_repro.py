"""
Regression script for a fixed segfault in concurrent forward-kinematics composition.

Not a test module (pytest never collects it): it is run in its own subprocess by
test_forward_kinematics_thread_safety.py, because the failure it used to reproduce was a
native crash that would take the whole pytest process down if triggered in-process.

The crash used to happen because ``World.compose_forward_kinematics_expression`` is
``@copy_memoize``: concurrent cache misses for the same key raced to both compute and
write the per-owner cache with no lock (krrood.patterns.caching.copy_memoize), and two
threads could end up calling ``deepcopy`` on the same cached CasADi-backed object at the
same time. The computation also built HomogeneousTransformationMatrix results via
SymbolicMathType.from_casadi_sx(), which used to alias rather than copy its input, and
HomogeneousTransformationMatrix._verify_type() writes into that casadi_sx in place. With
enough concurrent callers, two threads ended up reading and writing overlapping CasADi
SX objects with no synchronization anywhere in that chain, which corrupted CasADi's
internal (non-atomic) reference counting and crashed natively. ``memoize``/
``copy_memoize`` now serialise each cached result, and the aliasing call sites now copy,
so this script is expected to run to completion without crashing.
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time

from semantic_digital_twin.adapters.urdf import URDFParser

APARTMENT_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "urdf",
    "apartment.urdf",
)


def hammer_forward_kinematics_composition(
    thread_count: int, duration_seconds: float
) -> None:
    world = URDFParser.from_file(file_path=APARTMENT_URDF_PATH).create_world()
    bodies = world.kinematic_structure_entities
    root = world.root
    stop_time = time.monotonic() + duration_seconds

    def worker(thread_index: int) -> None:
        random_generator = random.Random(thread_index)
        while time.monotonic() < stop_time:
            body_a, body_b = random_generator.sample(bodies, 2)
            world.compose_forward_kinematics_expression(body_a, body_b)
            world.compose_forward_kinematics_expression(root, body_a)
            world.compose_forward_kinematics_expression(root, body_b)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-count", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    arguments = parser.parse_args()
    hammer_forward_kinematics_composition(
        thread_count=arguments.thread_count, duration_seconds=arguments.duration_seconds
    )
