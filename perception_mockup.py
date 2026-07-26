"""Minimal perception mockup for the cup-to-cup transfer.

Publishes a noisy measurement of the receiving cup's fill level so the Giskard server applies it to
the fill DOF mid-motion. Combined with the server's own model integration this gives
predictor-corrector behaviour: the model advances the fill between measurements, and each
measurement re-anchors the belief around the (noisy) truth.

Run order (three terminals):
  1. tracy_standalone.py       -- the Giskard server
  2. demo_pouring_transfer.py  -- adds the cups, flags the receiver fill DOF, starts the pour
  3. perception_mockup.py      -- this script

The mockup is generic: it drives whichever DOF the world model flags as externally updatable
(``DegreeOfFreedom.allows_external_state_update``), so it needs no knowledge of the cup itself.
"""

import atexit
import random
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# ---- Parameters ----
PERCEPTION_HZ = 5.0
"""Rate at which measurements are published, well below the 80 Hz control rate."""

NOISE_SIGMA = 0.02
"""Standard deviation of the zero-mean Gaussian measurement noise, in normalized fill units."""


def _find_externally_updatable_dof(world: World) -> DegreeOfFreedom | None:
    """Return the first DOF the world flags as externally updatable, or ``None`` if none exists yet."""
    for dof in world.degrees_of_freedom:
        if dof.allows_external_state_update:
            return dof
    return None


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("perception_mockup")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(
        target=executor.spin, daemon=True, name="perception-mockup-spin"
    ).start()

    # Re-fetch a fresh snapshot until the transfer demo has added and flagged the receiver fill DOF.
    # Fetching (rather than subscribing and waiting) gets the complete current model, so we never
    # start applying live state updates against a world that is missing the cups -- which would
    # otherwise raise StateUpdateContainsUnknownDegreesOfFreedomError.
    print("Waiting for an externally-updatable DOF (added by the transfer demo)...")
    fill_dof = None
    while fill_dof is None:
        world = fetch_world_from_service(node, timeout_seconds=300)
        fill_dof = _find_externally_updatable_dof(world)
        if fill_dof is None:
            time.sleep(0.5)

    # Only now go live: the snapshot already contains every DOF, so incoming state updates are safe.
    synchronizer = WorldSynchronizer(_world=world, node=node)
    atexit.register(synchronizer.close)
    print(
        f"Publishing noisy fill measurements for {fill_dof.name} at {PERCEPTION_HZ} Hz."
    )

    period = 1.0 / PERCEPTION_HZ
    while rclpy.ok():
        true_fill = float(world.state[fill_dof.id].position)
        perceived_fill = min(1.0, max(0.0, true_fill + random.gauss(0.0, NOISE_SIGMA)))
        world.state[fill_dof.id].position = perceived_fill
        world.notify_state_change()
        time.sleep(period)


if __name__ == "__main__":
    main()
