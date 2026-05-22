from __future__ import annotations

import os
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from typing_extensions import Dict, Optional

from semantic_digital_twin.adapters.rerun import (
    MeshFileResolver,
    RerunSink,
    RerunVisualizer,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.world_entity import Body

UR_TYPE = "ur5"
"""Universal Robots model whose local meshes are used for color (must match the publisher)."""

UR_XACRO = os.path.join(
    get_package_share_directory("ur_description"),
    "urdf",
    "ur.urdf.xacro",
)
"""Path to the UR description xacro (requires the ``ur_description`` package)."""


def load_example_world() -> World:
    """Parse the UR robot xacro into a world.

    :return: The populated world.
    """
    return URDFParser.from_xacro(
        UR_XACRO, mappings={"ur_type": UR_TYPE, "name": UR_TYPE}
    ).parse()


def build_local_mesh_resolver() -> MeshFileResolver:
    """Build a resolver that recovers mesh color from local files.

    The mirrored geometry arrives without materials, so color must come from the
    original mesh files. This parses the same robot description locally and maps
    each body to its mesh file, which the adapter then loads for full color.
    Bodies without a matching local mesh fall back to the flat mirrored geometry.

    :return: A resolver mapping a body to its original local mesh file path.
    """
    reference_world = load_example_world()
    mesh_by_body_name: Dict[str, str] = {}
    for body in reference_world.bodies:
        for shape in body.visual.shapes:
            if isinstance(shape, Mesh):
                mesh_by_body_name[body.name.name] = shape.filename
                break

    def resolver(body: Body, mesh: Mesh) -> Optional[str]:
        return mesh_by_body_name.get(body.name.name)

    return resolver


def main() -> None:
    """Mirror the published world and render it in Rerun until interrupted."""
    rclpy.init()
    node = rclpy.create_node("semdt_viewer")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    world = fetch_world_from_service(node)
    StateSynchronizer(_world=world, node=node)
    ModelSynchronizer(_world=world, node=node)
    RerunVisualizer(
        _world=world,
        application_id="semdt_rerun_viewer",
        sink=RerunSink.SPAWN,
        state_history=False,
        memory_limit="1GB",
        mesh_file_resolver=build_local_mesh_resolver(),
    )

    print("Viewer running; updates follow the publisher. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
