"""
Generate the items the robot transports through the chemical laboratory.

Produces into ``meshes/`` next to this script:

* ``sample_flask.stl`` — an Erlenmeyer flask, the sample coming in for analysis.
* ``reagent_bottle.stl`` — a capped cylindrical reagent bottle.
* ``test_tube_rack.stl`` — a small rack with four test tubes, the empties going
  back to the preparation bench.

Every mesh is centered on its bounding-box center, so a spawn pose of *surface
height + half of the mesh's z extent* rests it on a surface. All of them are
narrow enough for the HSR's parallel gripper.

Run once (the generated files are checked in)::

    python generate_laboratory_items.py
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

MESH_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
"""
Where the generated items land.
"""

REVOLVE_SECTIONS = 48
"""
Circular resolution of the revolved glassware.
"""

# %% glassware profiles


def revolved_solid(profile: list[tuple[float, float]]) -> trimesh.Trimesh:
    """
    :param profile: The (radius, height) outline of a rotationally symmetric body,
        starting and ending on the rotation axis.
    :return: The solid of revolution around the z axis.
    """
    return trimesh.creation.revolve(
        np.array(profile, dtype=float), sections=REVOLVE_SECTIONS
    )


def sample_flask() -> trimesh.Trimesh:
    """
    :return: An Erlenmeyer flask, 11.5 cm tall with a 7 cm base and a 2.8 cm neck.
    """
    return revolved_solid(
        [
            (0.0, 0.0),
            (0.035, 0.0),
            (0.014, 0.085),
            (0.014, 0.115),
            (0.0, 0.115),
        ]
    )


def reagent_bottle() -> trimesh.Trimesh:
    """
    :return: A capped reagent bottle, 15 cm tall with a 6 cm body.
    """
    return revolved_solid(
        [
            (0.0, 0.0),
            (0.030, 0.0),
            (0.030, 0.100),
            (0.016, 0.115),
            (0.016, 0.130),
            (0.0185, 0.130),
            (0.0185, 0.150),
            (0.0, 0.150),
        ]
    )


# %% test tube rack


def test_tube_rack() -> trimesh.Trimesh:
    """
    :return: A four-slot rack, 12 x 6 x 9 cm, its tubes standing upright through the
        top plate.
    """
    base = trimesh.creation.box(bounds=[[-0.06, -0.03, 0.0], [0.06, 0.03, 0.012]])
    top_plate = trimesh.creation.box(bounds=[[-0.06, -0.03, 0.058], [0.06, 0.03, 0.07]])
    end_walls = [
        trimesh.creation.box(bounds=[[x, -0.03, 0.0], [x + 0.008, 0.03, 0.07]])
        for x in (-0.06, 0.052)
    ]
    tubes = []
    for tube_x in np.linspace(-0.036, 0.036, 4):
        tube = trimesh.creation.cylinder(radius=0.008, height=0.078, sections=24)
        tube.apply_translation([tube_x, 0.0, 0.012 + 0.039])
        tubes.append(tube)
    return trimesh.util.concatenate([base, top_plate, *end_walls, *tubes])


# %% mesh export

ITEM_BUILDERS = {
    "sample_flask.stl": sample_flask,
    "reagent_bottle.stl": reagent_bottle,
    "test_tube_rack.stl": test_tube_rack,
}
"""
The items of the demo and how each is built.
"""

os.makedirs(MESH_DIRECTORY, exist_ok=True)
for file_name, builder in ITEM_BUILDERS.items():
    mesh = builder()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.export(os.path.join(MESH_DIRECTORY, file_name))
    extents = mesh.bounding_box.extents
    print(
        "%s: extents=(%.4f, %.4f, %.4f) half_height=%.4f"
        % (file_name, extents[0], extents[1], extents[2], extents[2] / 2)
    )
