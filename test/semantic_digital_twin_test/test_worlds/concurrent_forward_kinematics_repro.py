"""
Regression script for a fixed segfault in concurrent forward-kinematics composition.

Not a test module: test_forward_kinematics_thread_safety.py runs it in its own
subprocess, because the crash it reproduced took the whole process down.
"""

from __future__ import annotations

import argparse
import random
import threading
import time

from semantic_digital_twin.adapters.urdf import URDFParser

APARTMENT_URDF_PATH = "package://iai_apartment/urdf/apartment.urdf"


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
