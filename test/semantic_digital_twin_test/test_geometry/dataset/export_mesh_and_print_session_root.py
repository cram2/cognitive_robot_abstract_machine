"""
Exports a mesh into the session root and prints that root, so a caller can observe what
the exporting process left behind after it exited.
"""

import trimesh

from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileStorage


def main() -> None:
    """
    Export one mesh with no explicit directory and print the session root it landed in.
    """
    Mesh.from_trimesh(mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    print(MeshFileStorage().root)


if __name__ == "__main__":
    main()
