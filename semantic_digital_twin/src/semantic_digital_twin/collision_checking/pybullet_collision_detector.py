import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict
from typing import List, Tuple, Optional
from uuid import UUID

import giskardpy_bullet_bindings as bullet
import numpy as np
import trimesh
from platformdirs import user_cache_dir
from trimesh import Trimesh

from giskardpy.utils.utils import create_path
from .collision_detector import (
    CollisionDetector,
    CollisionCheckingResult,
    ClosestPoints,
)
from .collision_matrix import CollisionMatrix
from ..utils import suppress_stdout_stderr
from ..world_description.geometry import (
    Shape,
    Box,
    Sphere,
    Cylinder,
    Scale,
    TriangleMesh,
    FileMesh,
    Mesh,
)
from ..world_description.world_entity import Body

logger = logging.getLogger(__name__)


def create_cache_dir(folder_name: str) -> Path:
    pkg_name = __package__.split(".", 1)[0]

    cache_dir = Path(user_cache_dir(pkg_name)) / folder_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


CACHE_DIR = create_cache_dir("convex_decompositions")
LOG_DIR = create_cache_dir("log")


def trimesh_quantized_hash(mesh, decimals: int = 6, digest_size: int = 16) -> str:
    """
    Hash tolerant to tiny float differences by rounding vertices.
    Still order-sensitive (vertex/face order changes -> different hash).
    """
    h = hashlib.blake2b(digest_size=digest_size)

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.faces)

    vq = np.round(v, decimals=decimals)
    vq = np.ascontiguousarray(vq)

    h.update(str(vq.shape).encode("utf-8"))
    h.update(str(vq.dtype).encode("utf-8"))
    h.update(vq.tobytes())

    h.update(str(f.shape).encode("utf-8"))
    h.update(str(f.dtype).encode("utf-8"))
    h.update(f.tobytes())

    return h.hexdigest()


def create_cube_shape(extents: Tuple[float, float, float]) -> bullet.BoxShape:
    """
    Creates a bullet box shape.
    :param extents: the extents of the box along x, y and z axis respectively.
    :return: the bullet box shape.
    """
    out = (
        bullet.BoxShape(bullet.Vector3(*[extents[x] * 0.5 for x in range(3)]))
        if type(extents) is not bullet.Vector3
        else bullet.BoxShape(extents)
    )
    out.margin = 0.001
    return out


def create_cylinder_shape(diameter: float, height: float) -> bullet.CylinderShape:
    """
    Creates a bullet cylinder shape.
    :param diameter: the diameter of the cylinder.
    :param height: the height of the cylinder.
    :return: the bullet cylinder shape.
    """
    out = bullet.CylinderShapeZ(bullet.Vector3(diameter / 2, diameter / 2, height))
    out.margin = 0.001
    return out


def create_sphere_shape(diameter: float) -> bullet.SphereShape:
    """
    Creates a bullet sphere shape.
    :param diameter: the diameter of the sphere.
    :return: the bullet sphere shape.
    """
    out = bullet.SphereShape(0.5 * diameter)
    out.margin = 0.001
    return out


def create_shape_from_geometry(geometry: Shape) -> bullet.CollisionShape:
    """
    Creates a bullet collision shape from a geometry.
    :param geometry: the geometry to create a collision shape from.
    :return: the bullet collision shape.
    """
    match geometry:
        case Box():
            shape = create_cube_shape(
                (geometry.scale.x, geometry.scale.y, geometry.scale.z)
            )

        case Sphere():
            shape = create_sphere_shape(geometry.radius * 2)

        case Cylinder():
            shape = create_cylinder_shape(
                diameter=geometry.width, height=geometry.height
            )

        case TriangleMesh():
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
            with open(f.name, "w") as fd:
                fd.write(trimesh.exchange.obj.export_obj(geometry.mesh))
            shape = load_convex_mesh_shape(
                mesh=geometry,
                single_shape=False,
                scale=geometry.scale,
            )

        case FileMesh():
            shape = load_convex_mesh_shape(
                mesh=geometry,
                single_shape=False,
                scale=geometry.scale,
            )

        case _:
            raise NotImplementedError()

    return shape


def create_shape_from_body(body: Body) -> bullet.CollisionObject:
    """
    Creates a bullet collision object from a body.
    :param body: the body to create a collision object from.
    :return: the bullet collision object.
    """
    shapes = []
    for collision_id, geometry in enumerate(body.collision):
        shape = create_shape_from_geometry(geometry=geometry)
        link_T_geometry = bullet.Transform.from_np(geometry.origin.to_np())
        shapes.append((link_T_geometry, shape))
    compouned_shape = create_compound_shape(shapes_poses=shapes)
    return create_object(body.id, compouned_shape, bullet.Transform.identity())


def create_compound_shape(
    shapes_poses: List[Tuple[bullet.Transform, bullet.CollisionShape]] = None,
) -> bullet.CompoundShape:
    """
    Creates a bullet compound shape.
    :param shapes_poses: the shapes and their poses.
    :return: the bullet compound shape.
    """
    out = bullet.CompoundShape()
    for t, s in shapes_poses:
        out.add_child(t, s)
    return out


def load_convex_mesh_shape(
    mesh: Mesh, single_shape: bool, scale: Scale
) -> bullet.ConvexShape:
    """
    Loads a convex mesh shape from a mesh.
    :param mesh: the mesh to load the convex shape from.
    :param single_shape: whether to load the mesh as a single shape.
    :param scale: the scale of the mesh.
    :return: the bullet convex shape.
    """
    if not mesh.mesh.is_convex:
        obj_pkg_filename = convert_to_decomposed_obj_and_save_in_tmp(mesh=mesh.mesh)
    else:
        obj_pkg_filename = mesh.filename
    return bullet.load_convex_shape(
        obj_pkg_filename,
        single_shape=single_shape,
        scaling=bullet.Vector3(scale.x, scale.y, scale.z),
    )


def clear_cache(cache_dir: Path = CACHE_DIR):
    """
    Clears the convex decomposition cache.
    :param cache_dir: the cache directory to clear.
    """
    for file in cache_dir.iterdir():
        file.unlink()


def convert_to_decomposed_obj_and_save_in_tmp(
    mesh: Trimesh,
    cache_dir: Path = CACHE_DIR,
    log_path: Path = LOG_DIR,
) -> str:
    """
    Converts a mesh to a convex decomposition and saves it in a cache file.
    :param mesh: the mesh to convert.
    :param cache_dir: the cache directory to save the convex decomposition in.
    :param log_path: the path to the log file.
    :return: the path to the convex decomposition file.
    """
    file_hash = trimesh_quantized_hash(mesh)
    obj_file_name = str(cache_dir / f"{file_hash}.obj")
    if not os.path.exists(obj_file_name):
        obj_str = trimesh.exchange.obj.export_obj(mesh)
        create_path(obj_file_name)
        with open(obj_file_name, "w") as f:
            f.write(obj_str)
        if not mesh.is_convex:
            with suppress_stdout_stderr():
                bullet.vhacd(obj_file_name, obj_file_name, str(log_path / "vhacd.log"))
            logging.info(f'Saved convex decomposition to "{obj_file_name}".')
        else:
            logging.info(f'Saved obj to "{obj_file_name}".')
    else:
        logging.debug(f'Cache hit, loaded "{obj_file_name}".')
    return obj_file_name


def create_object(
    name: UUID,
    shape: bullet.CollisionShape,
    transform: Optional[bullet.Transform] = None,
) -> bullet.CollisionObject:
    """
    Creates a bullet collision object.
    :param name: the name of the collision object.
    :param shape: the collision shape of the collision object.
    :param transform: the transform of the collision object.
    :return: the bullet collision object.
    """
    if transform is None:
        transform = bullet.Transform.identity()
    out = bullet.CollisionObject(name)
    out.collision_shape = shape
    out.collision_flags = bullet.CollisionObject.KinematicObject
    out.transform = transform
    return out


@dataclass(eq=False)
class BulletCollisionDetector(CollisionDetector):
    """
    Collision detector based on the giskard wrapper for bullet.
    """

    kineverse_world: bullet.KineverseWorld = field(
        default_factory=bullet.KineverseWorld, init=False
    )
    """
    Reference to the bullet world.
    """
    body_to_bullet_object: Dict[Body, bullet.CollisionObject] = field(
        default_factory=dict, init=False
    )
    """
    Maps semdt world bodies to their bullet collision objects.
    """
    _ordered_bullet_objects: List[bullet.CollisionObject] = field(
        default_factory=list, init=False
    )
    """
    The bullet collision objects in the order they are added to the world.
    This is only a cache for performance reasons.
    """

    def sync_world_model(self) -> None:
        self.reset_cache()
        self.clear()
        self.body_to_bullet_object = {}
        if self._world.is_empty():
            return
        for body in self._world.bodies_with_collision:
            self.add_body(body)
        self._ordered_bullet_objects = list(self.body_to_bullet_object.values())

    def clear(self):
        for o in self.kineverse_world.collision_objects:
            self.kineverse_world.remove_collision_object(o)

    def sync_world_state(self) -> None:
        if len(self._ordered_bullet_objects) > 0:
            bullet.batch_set_transforms(
                self._ordered_bullet_objects,
                self.get_all_collision_fks(),
            )

    def add_body(self, body: Body):
        o = create_shape_from_body(body=body)
        self.kineverse_world.add_collision_object(o)
        self.body_to_bullet_object[body] = o

    def reset_cache(self):
        self.collision_matrix_to_bullet_query.cache_clear()

    def __hash__(self):
        return hash(id(self))

    @lru_cache(maxsize=100)
    def collision_matrix_to_bullet_query(
        self, collision_matrix: CollisionMatrix
    ) -> Optional[Dict[Tuple[bullet.CollisionObject, bullet.CollisionObject], float]]:
        return {
            (
                self.body_to_bullet_object[check.body_a],
                self.body_to_bullet_object[check.body_b],
            ): check.distance
            for check in collision_matrix.collision_checks
        }

    def check_collisions(
        self, collision_matrix: CollisionMatrix
    ) -> CollisionCheckingResult:

        query = self.collision_matrix_to_bullet_query(collision_matrix)
        result: List[bullet.Collision] = (
            self.kineverse_world.get_closest_filtered_map_batch(query)
        )
        return CollisionCheckingResult(
            [
                ClosestPoints(
                    body_a=self._world.get_kinematic_structure_entity_by_id(
                        collision.obj_a.name
                    ),
                    body_b=self._world.get_kinematic_structure_entity_by_id(
                        collision.obj_b.name
                    ),
                    distance=collision.contact_distance,
                    root_P_point_on_body_a=collision.map_P_pa,
                    root_P_point_on_body_b=collision.map_P_pb,
                    root_V_contact_normal_from_b_to_a=collision.world_V_n,
                )
                for collision in result
            ]
        )
