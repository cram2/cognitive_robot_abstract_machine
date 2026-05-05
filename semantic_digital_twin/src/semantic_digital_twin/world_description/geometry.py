from __future__ import annotations

import itertools
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cached_property

from plyfile import PlyData
import numpy as np
import trimesh
import trimesh.exchange.stl
from PIL import Image
from trimesh.visual import ColorVisuals
from trimesh.visual.texture import TextureVisuals, SimpleMaterial
from typing_extensions import (
    Optional,
    List,
    Dict,
    Any,
    Self,
    Tuple,
    TYPE_CHECKING,
    assert_never,
)

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from random_events.interval import SimpleInterval, Bound, closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.mixin import HasSimulatorProperties
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.utils import IDGenerator

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

id_generator = IDGenerator()

logger = logging.getLogger(__name__)


@dataclass
class Color:
    """
    Dataclass for storing rgba_color as an RGBA value.
    The values are stored as floats between 0 and 1.
    The default rgba_color is white.
    """

    R: float = 1.0
    """
    Red value of the color.
    """

    G: float = 1.0
    """
    Green value of the color.
    """

    B: float = 1.0
    """
    Blue value of the color.
    """

    A: float = 1.0
    """
    Opacity of the color.
    """

    def __post_init__(self):
        """
        Make sure the color values are floats, because ros2 sucks.
        """
        self.R = float(self.R)
        self.G = float(self.G)
        self.B = float(self.B)
        self.A = float(self.A)

    def to_rgba(self) -> Tuple[float, float, float, float]:
        return (self.R, self.G, self.B, self.A)

    def to_rgb(self) -> Tuple[float, float, float]:
        return (self.R, self.G, self.B)

    @classmethod
    def from_list(cls, color: List[float]):
        """
        Set the rgba_color from a list of RGBA values.

        :param color: The list of RGBA values
        """
        if len(color) == 3:
            return cls.from_rgb(color)
        elif len(color) == 4:
            return cls.from_rgba(color)
        else:
            raise ValueError("Color list must have 3 or 4 elements")

    @classmethod
    def from_rgb(cls, rgb: List[float]):
        """
        Set the rgba_color from a list of RGB values.

        :param rgb: The list of RGB values
        """
        return cls(*rgb, 1)

    @classmethod
    def from_rgba(cls, rgba: List[float]):
        """
        Set the rgba_color from a list of RGBA values.

        :param rgba: The list of RGBA values
        """
        return cls(*rgba)

    @classmethod
    def PINK(cls) -> Self:
        return cls(1, 0, 1, 1)

    @classmethod
    def BLACK(cls) -> Self:
        return cls(0, 0, 0, 1)

    @classmethod
    def WHITE(cls) -> Self:
        return cls(1, 1, 1, 1)

    @classmethod
    def RED(cls) -> Self:
        return cls(1, 0, 0, 1)

    @classmethod
    def GREEN(cls) -> Self:
        return cls(0, 1, 0, 1)

    @classmethod
    def BLUE(cls) -> Self:
        return cls(0, 0, 1, 1)

    @classmethod
    def YELLOW(cls) -> Self:
        return cls(1, 1, 0, 1)

    @classmethod
    def CYAN(cls) -> Self:
        return cls(0, 1, 1, 1)

    @classmethod
    def MAGENTA(cls) -> Self:
        return cls(1, 0, 1, 1)

    @classmethod
    def GREY(cls) -> Self:
        return cls(0.5, 0.5, 0.5, 1)


@dataclass
class Scale:
    """
    Dataclass for storing the scale of geometric objects.
    """

    x: float = 1.0
    """
    The scale in the x direction.
    """

    y: float = 1.0
    """
    The scale in the y direction.
    """

    z: float = 1.0
    """
    The scale in the z direction.
    """

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __post_init__(self):
        """
        Make sure the scale values are floats, because ros2 sucks.
        """
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)

    def to_simple_event(
        self,
        extend_result_in_direction: Optional[Vector3] = None,
        amount: float = 0.0,
    ) -> SimpleEvent:
        simple_event = SimpleEvent.from_data(
            {
                SpatialVariables.x.value: closed(-self.x / 2, self.x / 2),
                SpatialVariables.y.value: closed(-self.y / 2, self.y / 2),
                SpatialVariables.z.value: closed(-self.z / 2, self.z / 2),
            }
        )

        if extend_result_in_direction is not None:
            self._extend_simple_event_in_direction(
                simple_event, extend_result_in_direction, amount
            )

        return simple_event

    def _extend_simple_event_in_direction(
        self, simple_event: SimpleEvent, direction: Vector3, amount: float
    ) -> SimpleEvent:
        """
        Extend the inner event in the specified direction to create the container opening in that direction.


        :return: The modified inner event with the specified direction extended.
        """
        match direction.to_np().tolist():
            case [1, 0, 0, 0]:
                simple_event[SpatialVariables.x.value] = closed(
                    -self.x / 2, self.x / 2 + amount
                )
            case [0, 1, 0, 0]:
                simple_event[SpatialVariables.y.value] = closed(
                    -self.y / 2, self.y / 2 + amount
                )
            case [0, 0, 1, 0]:
                simple_event[SpatialVariables.z.value] = closed(
                    -self.z / 2, self.z / 2 + amount
                )
            case [-1, 0, 0, 0]:
                simple_event[SpatialVariables.x.value] = closed(
                    -(self.x / 2 + amount), self.x / 2
                )
            case [0, -1, 0, 0]:
                simple_event[SpatialVariables.y.value] = closed(
                    -(self.y / 2 + amount), self.y / 2
                )
            case [0, 0, -1, 0]:
                simple_event[SpatialVariables.z.value] = closed(
                    -(self.z / 2 + amount), self.z / 2
                )

        return simple_event

    def to_bounding_box(self) -> BoundingBox:
        min_point = Point3(-self.x / 2, -self.y / 2, -self.z / 2)
        max_point = Point3(self.x / 2, self.y / 2, self.z / 2)
        return BoundingBox.from_min_max(min_point, max_point, None)

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Shape(ABC, SubclassJSONSerializer, HasSimulatorProperties):
    """
    Base class for all shapes in the world.
    """

    origin: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )

    color: Color = field(default_factory=Color)

    @property
    @abstractmethod
    def local_frame_bounding_box(self) -> BoundingBox:
        """
        Returns the bounding box of the shape
        """

    @property
    @abstractmethod
    def mesh(self) -> trimesh.Trimesh:
        """
        The mesh object of the shape.
        This should be implemented by subclasses.
        """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "origin": to_json(self.origin),
            "color": to_json(self.color),
        }

    def __eq__(self, other: Shape) -> bool:
        """Custom equality comparison that handles TransformationMatrix equivalence"""
        if not isinstance(other, self.__class__):
            return False

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(self)]

        for field_name in field_names:
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)

            if field_name != "origin":
                if self_value != other_value:
                    return False
        if not np.allclose(self.origin.to_np(), other.origin.to_np()):
            return False

        return True

    def copy_for_world(self, world: World) -> Self:
        """
        Copies this shape with references to the given world.
        :param world: The world to copy to.
        :return: A copy of this shape with references to the given world.
        """
        new_origin = HomogeneousTransformationMatrix(
            self.origin.to_np(),
        )
        shape_props = fields(self)
        new_props = {
            f.name: deepcopy(getattr(self, f.name))
            for f in shape_props
            if f.name not in ["origin"]
        }
        return self.__class__(origin=new_origin, **new_props)


@dataclass(eq=False)
class Mesh(Shape):
    """
    Abstract mesh class.
    Subclasses must provide a `mesh` property returning a trimesh.Trimesh.
    """

    scale: Scale = field(default_factory=Scale)
    """
    Scale of the mesh.
    """

    filename: str = ""
    """
    Filename of the mesh.
    """

    @property
    def local_frame_bounding_box(self) -> BoundingBox:
        """
        Returns the local bounding box of the mesh.
        The bounding box is axis-aligned and centered at the origin.
        """
        return BoundingBox.from_mesh(self.mesh, self.origin)

    def to_json(self) -> Dict[str, Any]:
        # Serialize the raw (unscaled, unprocessed) mesh geometry and the scale separately
        base_mesh = trimesh.load_mesh(self.filename, process=False)
        file_type = self.filename.split(".")[-1]
        return {
            **super().to_json(),
            "mesh": base_mesh.to_dict(),
            "scale": to_json(self.scale),
            "file_type": file_type,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Mesh:
        # Recreate the trimesh without processing to preserve exact topology
        mesh = trimesh.Trimesh(
            vertices=data["mesh"]["vertices"],
            faces=data["mesh"]["faces"],
            process=False,
        )
        # TODO correctly sync color/texture of DAE
        if True or mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([1.0, 1.0, 1.0]).reshape(1, 3), (len(mesh.vertices), 1)
            )

        origin = from_json(data["origin"], **kwargs)
        scale = from_json(data["scale"], **kwargs)
        file_type = data["file_type"]
        return cls.from_trimesh(
            mesh=mesh, origin=origin, scale=scale, file_type=file_type
        )

    @classmethod
    def add_uv(cls, mesh: trimesh.Trimesh, uv: np.ndarray) -> trimesh.Trimesh:
        faces = mesh.faces
        vertices = mesh.vertices
        # 1. Expand vertices so each face corner gets its own vertex
        vertex_indices_expanded = faces.reshape(-1)  # (F*3,)
        vertices_new = vertices[vertex_indices_expanded]  # (F*3, 3)

        # 2. New faces are just 0..F*3-1 reshaped into triples
        faces_new = np.arange(len(vertices_new), dtype=np.int64).reshape(-1, 3)

        # 3. Create mesh with expanded vertices
        mesh = trimesh.Trimesh(vertices=vertices_new, faces=faces_new, process=False)
        mesh.visual = TextureVisuals(uv=uv)
        return mesh

    @classmethod
    def add_texture(
        cls, mesh: trimesh.Trimesh, texture_file_path: str
    ) -> trimesh.Trimesh:
        image = Image.open(texture_file_path)
        material_name = os.path.splitext(os.path.basename(texture_file_path))[0]
        mesh.visual.material = SimpleMaterial(name=material_name, image=image)
        return mesh

    def scale_mesh(self, scale: Scale) -> trimesh.Trimesh:
        """
        Scales the mesh according to the given scale.

        :param scale: The scale of the mesh.
        :return: A scaled mesh object.
        """
        copy_mesh = deepcopy(self.mesh)
        copy_mesh.apply_scale(scale.to_np())
        return copy_mesh

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """
        The mesh object.
        """
        mesh = trimesh.load_mesh(self.filename)
        mesh.apply_scale(self.scale.to_np())
        mesh.visual.vertex_colors = trimesh.visual.color.to_rgba(self.color.to_rgba())
        return mesh

    @classmethod
    def from_file(
        cls, file_path: str, texture_file_path: Optional[str] = None, **kwargs
    ) -> Mesh:
        """
        Create a Mesh from a file path.

        :param file_path: Path to the mesh file.
        :param texture_file_path: Optional path to the texture file.
        :return: Mesh object.
        """
        file_mesh = cls(filename=file_path, **kwargs)
        if texture_file_path is not None:
            file_mesh.mesh = cls.add_texture(
                mesh=file_mesh.mesh, texture_file_path=texture_file_path
            )
        return file_mesh

    @classmethod
    def from_ply_file(
        cls,
        ply_file_path: str,
        texture_file_path: Optional[str] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        scale: Optional[Scale] = None,
    ) -> Mesh:
        """
        Create a Mesh from a PLY file path and an optional texture file path.
        Ply files are not supported by RViz2, so we need to convert them to OBJ files with the textures intact.
        """
        texture_image = Image.open(texture_file_path)
        ply_file = PlyData.read(ply_file_path)
        # Raw data
        vertices = np.stack(
            [ply_file["vertex"]["x"], ply_file["vertex"]["y"], ply_file["vertex"]["z"]],
            axis=-1,
        )

        texture_coordinates = np.stack(
            [ply_file["texcoord"]["s"], ply_file["texcoord"]["t"]], axis=-1
        )

        faces = np.stack([np.array(f["vertex_indices"]) for f in ply_file["face"]])
        texture_coordinate_indices = np.stack(
            [np.array(f["texcoord_indices"]) for f in ply_file["face"]]
        )

        # Build per-vertex UV by unpacking face-corner UVs
        # texture_coordinate_indices[f, c] -> index into texture_coordinates for face f, corner c
        uv_per_corner = texture_coordinates[texture_coordinate_indices]

        vertices_unindexed = vertices[faces.reshape(-1)]
        uv_unindexed = uv_per_corner.reshape(-1, 2)
        faces_new = np.arange(len(vertices_unindexed)).reshape(-1, 3)

        mesh = trimesh.Trimesh(
            vertices=vertices_unindexed,
            faces=faces_new,
            visual=trimesh.visual.TextureVisuals(uv=uv_unindexed, image=texture_image),
        )

        return Mesh.from_trimesh(mesh=mesh, origin=origin, scale=scale, file_type="obj")

    @classmethod
    def from_trimesh(
        cls,
        mesh: trimesh.Trimesh,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        scale: Optional[Scale] = None,
        uv: Optional[np.ndarray] = None,
        texture_file_path: Optional[str] = None,
        dirname: str = "/tmp",
        file_type: str = "obj",
    ) -> "Mesh":
        file_type = file_type.lower()
        if origin is None:
            origin = HomogeneousTransformationMatrix()
        if scale is None:
            scale = Scale()
        if uv is not None:
            mesh = cls.add_uv(mesh=mesh, uv=uv)
        if texture_file_path is not None:
            mesh = cls.add_texture(mesh=mesh, texture_file_path=texture_file_path)

        # Each export gets its own subdir so material.mtl files never collide
        subdir = tempfile.mkdtemp(dir=dirname)
        tmp_path = os.path.join(subdir, f"mesh.{file_type}")

        try:
            mesh.export(tmp_path, file_type=file_type)
        except Exception:
            shutil.rmtree(subdir, ignore_errors=True)
            raise

        instance = cls(
            origin=origin,
            scale=scale,
            filename=tmp_path,
        )

        # # Tie file lifetime to the Mesh instance TODO luca wants to find a way for this to work with rviz (atexit)
        # weakref.finalize(instance, cls._cleanup_temp_dir, subdir)

        return instance

    @staticmethod
    def _cleanup_temp_dir(subdir: str) -> None:
        """
        Clean up the temporary subdirectory created for the mesh.
        """
        logger.debug(f"Cleaning up temporary directory: {subdir}")
        try:
            shutil.rmtree(subdir, ignore_errors=True)
        except OSError:
            pass

    @classmethod
    def from_vertices_and_faces(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        origin: np.ndarray,
        scale: np.ndarray,
        uv: Optional[np.ndarray] = None,
        texture_file_path: Optional[str] = None,
    ) -> Mesh:
        """
        Create a mesh from vertices, faces, origin, and scale.

        :param vertices: Vertices of the mesh.
        :param faces: Faces of the mesh.
        :param origin: Origin of the mesh.
        :param scale: Scale of the mesh.
        :param uv: Optional UV coordinates.
        :return: Mesh object.
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return cls.from_trimesh(
            mesh=mesh,
            origin=origin,
            scale=scale,
            uv=uv,
            texture_file_path=texture_file_path,
        )

    @classmethod
    def from_3d_points(
        cls,
        points_3d: List[Point3],
        reference_frame: Optional[KinematicStructureEntity] = None,
        minimum_thickness: float = 0.005,
        sv_ratio_tol: float = 1e-7,
    ) -> Self:
        """
        Constructs a Region from a list of 3D points by creating a convex hull around them.
        The points are analyzed to determine if they are approximately planar. If they are,
        a minimum thickness is added to ensure the region has a non-zero volume.

        :param name: Prefixed name for the region.
        :param points_3d: List of 3D points.
        :param reference_frame: Optional reference frame.
        :param minimum_thickness: Minimum thickness to add if points are near-planar.
        :param sv_ratio_tol: Tolerance for determining planarity based on singular value ratio.

        :return: Region object.
        """
        points = np.asarray([point.to_np()[:3] for point in points_3d], dtype=float)
        points = np.unique(points, axis=0)
        assert (
            len(points) >= 3
        ), "At least 4 unique points are required to define a 3D region."

        centered_points = points - points.mean(axis=0, keepdims=True)
        assert np.any(centered_points), "Points must not be all identical."

        # We compute the principal axes of the point cloud using SVD.
        # This allows us to reason about the geometric thickness of our point cloud.
        # The axis with the smallest variance, located at the last index if our `principal_axis` is our `normal`
        # indicating the direction of the region's thickness.
        _, variance, principal_axis = np.linalg.svd(
            centered_points, full_matrices=False
        )
        smallest_variance_axis = principal_axis[-1]  # this is our normal
        unit_vector_normal = smallest_variance_axis / np.linalg.norm(
            smallest_variance_axis
        )

        # We compute the thickness, peak-to-peak (max - min), along the normal direction, to get the thickness of
        # the region.
        thickness_in_normal_direction = np.ptp(centered_points @ unit_vector_normal)
        is_near_planar = variance[0] > 0 and variance[-1] / variance[0] < sv_ratio_tol
        thickness_padding = (
            minimum_thickness / 2
            if thickness_in_normal_direction < minimum_thickness or is_near_planar
            else 0.0
        )

        # We do not provide any 2d shapes, since they would be very weird to handle with raytracing etc.
        # Thus we decided that in near-planar cases we add a minimum thickness to ensure we get a 3d shape.
        if thickness_padding > 0:
            P_aug = np.vstack(
                [
                    centered_points + thickness_padding * unit_vector_normal,
                    centered_points - thickness_padding * unit_vector_normal,
                ]
            )
        else:
            P_aug = centered_points

        hull = trimesh.points.PointCloud(P_aug).convex_hull
        hull.remove_unreferenced_vertices()
        hull.update_faces(hull.nondegenerate_faces())
        hull.process()

        return cls.from_trimesh(
            mesh=hull,
            origin=HomogeneousTransformationMatrix(reference_frame=reference_frame),
        )

    @classmethod
    def project_texture_coordinates(
        cls,
        mesh: trimesh.Trimesh,
        projection_axis: np.ndarray,
        scale: np.ndarray,
    ) -> trimesh.Trimesh:
        """
        Generate texture coordinates by projecting vertices along an axis and normalizing by scale.
        This prepares the mesh for rendering with a texture map using
        `UV mapping <https://en.wikipedia.org/wiki/UV_mapping>`_.

        :param mesh: The mesh to apply UVs to.
        :param projection_axis: A (3,) array representing the axis to project along (e.g., [0, 0, 1] for Z).
        :param scale: A (3,) array for normalizing the UV coordinates (e.g., dimensions of the box).
        :return: A new mesh with UV coordinates and expanded vertices.
        """
        # Expand vertices for each face corner
        faces = mesh.faces.reshape(-1)
        vertices = mesh.vertices[faces]

        # Identify the two axes perpendicular to the projection axis
        # This is a simplified version for axis-aligned projections (X, Y, or Z)
        axes = np.where(projection_axis == 0)[0]
        if len(axes) != 2:
            # Fallback or more complex logic for non-axis-aligned projections could go here
            axes = [0, 1] if projection_axis[2] != 0 else [0, 2]

        # Calculate UVs by projecting and normalizing
        uv = np.zeros((len(vertices), 2))
        uv[:, 0] = (vertices[:, axes[0]] / scale[axes[0]]) + 0.5
        uv[:, 1] = (vertices[:, axes[1]] / scale[axes[1]]) + 0.5

        return cls.add_uv(mesh=mesh, uv=uv)


@dataclass(eq=False)
class Sphere(Shape):
    """
    A sphere shape.
    """

    radius: float = 0.5
    """
    Radius of the sphere.
    """

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Returns a trimesh object representing the sphere.
        """
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=self.radius)
        mesh.visual.vertex_colors = trimesh.visual.color.to_rgba(self.color.to_rgba())
        return mesh

    @property
    def local_frame_bounding_box(self) -> BoundingBox:
        """
        Returns the bounding box of the sphere.
        """
        return BoundingBox(
            -self.radius,
            -self.radius,
            -self.radius,
            self.radius,
            self.radius,
            self.radius,
            self.origin,
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "radius": self.radius}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            radius=data["radius"],
            origin=from_json(data["origin"], **kwargs),
            color=from_json(data["color"], **kwargs),
        )


@dataclass(eq=False)
class Cylinder(Shape):
    """
    A cylinder shape.
    """

    width: float = 0.5
    height: float = 0.5

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Returns a trimesh object representing the cylinder.
        """
        mesh = trimesh.creation.cylinder(
            radius=self.width / 2, height=self.height, sections=16
        )
        mesh.visual.vertex_colors = trimesh.visual.color.to_rgba(self.color.to_rgba())
        return mesh

    @property
    def local_frame_bounding_box(self) -> BoundingBox:
        """
        Returns the bounding box of the cylinder.
        The bounding box is axis-aligned and centered at the origin.
        """
        half_width = self.width / 2
        half_height = self.height / 2
        return BoundingBox(
            -half_width,
            -half_width,
            -half_height,
            half_width,
            half_width,
            half_height,
            self.origin,
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "width": self.width, "height": self.height}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            width=data["width"],
            height=data["height"],
            origin=from_json(data["origin"], **kwargs),
            color=from_json(data["color"], **kwargs),
        )


@dataclass(eq=False)
class Box(Shape):
    """
    A box shape. Pivot point is at the center of the box.
    """

    scale: Scale = field(default_factory=Scale)

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Returns a trimesh object representing the box.
        The box is centered at the origin and has the specified scale.
        """
        mesh = trimesh.creation.box(extents=(self.scale.x, self.scale.y, self.scale.z))
        mesh.visual.vertex_colors = trimesh.visual.color.to_rgba(self.color.to_rgba())
        return mesh

    @property
    def local_frame_bounding_box(self) -> BoundingBox:
        """
        Returns the local bounding box of the box.
        The bounding box is axis-aligned and centered at the origin.
        """
        half_x = self.scale.x / 2
        half_y = self.scale.y / 2
        half_z = self.scale.z / 2
        return BoundingBox(
            -half_x,
            -half_y,
            -half_z,
            half_x,
            half_y,
            half_z,
            self.origin,
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "scale": to_json(self.scale)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=from_json(data["scale"], **kwargs),
            origin=from_json(data["origin"], **kwargs),
            color=from_json(data["color"], **kwargs),
        )


@dataclass(eq=False)
class BoundingBox:
    min_x: float
    """
    The minimum x-coordinate of the bounding box, relative to the origin.
    """

    min_y: float
    """
    The minimum y-coordinate of the bounding box, relative to the origin.
    """

    min_z: float
    """
    The minimum z-coordinate of the bounding box, relative to the origin.
    """

    max_x: float
    """
    The maximum x-coordinate of the bounding box, relative to the origin.
    """

    max_y: float
    """
    The maximum y-coordinate of the bounding box, relative to the origin.
    """

    max_z: float
    """
    The maximum z-coordinate of the bounding box, relative to the origin.
    """

    origin: HomogeneousTransformationMatrix
    """
    The origin of the bounding box.
    """

    def __hash__(self):
        # The hash should be this since comparing those via hash is checking if those are the same and not just equal
        return hash(
            (self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z)
        )

    @property
    def x_interval(self) -> SimpleInterval:
        """
        :return: The x interval of the bounding box.
        """
        return SimpleInterval.from_data(
            float(self.origin.x + self.min_x),
            float(self.origin.x + self.max_x),
            Bound.CLOSED,
            Bound.CLOSED,
        )

    @property
    def y_interval(self) -> SimpleInterval:
        """
        :return: The y interval of the bounding box.
        """
        return SimpleInterval.from_data(
            float(self.origin.y + self.min_y),
            float(self.origin.y + self.max_y),
            Bound.CLOSED,
            Bound.CLOSED,
        )

    @property
    def z_interval(self) -> SimpleInterval:
        """
        :return: The z interval of the bounding box.
        """
        return SimpleInterval.from_data(
            float(self.origin.z + self.min_z),
            float(self.origin.z + self.max_z),
            Bound.CLOSED,
            Bound.CLOSED,
        )

    @property
    def scale(self) -> Scale:
        """
        :return: The scale of the bounding box.
        """
        return Scale(self.depth, self.width, self.height)

    @property
    def depth(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def width(self) -> float:
        return self.max_y - self.min_y

    @property
    def simple_event(self) -> SimpleEvent:
        """
        :return: The bounding box as a random event.
        """
        return SimpleEvent.from_data(
            {
                SpatialVariables.x.value: self.x_interval,
                SpatialVariables.y.value: self.y_interval,
                SpatialVariables.z.value: self.z_interval,
            }
        )

    @property
    def dimensions(self) -> List[float]:
        """
        :return: The dimensions of the bounding box as a list [width, depth, height].
        """
        return [self.depth, self.width, self.height]

    def bloat(
        self, x_amount: float = 0.0, y_amount: float = 0, z_amount: float = 0
    ) -> BoundingBox:
        """
        Enlarges the bounding box by a given amount in all dimensions.

        :param x_amount: The amount to adjust minimum and maximum x-coordinates
        :param y_amount: The amount to adjust minimum and maximum y-coordinates
        :param z_amount: The amount to adjust minimum and maximum z-coordinates
        :return: New enlarged bounding box
        """
        return self.__class__(
            self.min_x - x_amount,
            self.min_y - y_amount,
            self.min_z - z_amount,
            self.max_x + x_amount,
            self.max_y + y_amount,
            self.max_z + z_amount,
            self.origin,
        )

    def contains(self, point: Point3) -> bool:
        """
        Check if the bounding box contains a point.
        """
        point_in_bb = point.reference_frame._world.transform(
            point, self.origin.reference_frame
        )
        x, y, z = (float(point_in_bb.x), float(point_in_bb.y), float(point_in_bb.z))
        return self.simple_event.contains((x, y, z))

    @classmethod
    def from_simple_event(
        cls, simple_event: SimpleEvent, origin: HomogeneousTransformationMatrix
    ) -> List[Self]:
        """
        Create a list of bounding boxes from a simple random event.

        :param simple_event: The random event.
        :param origin: The origin of the intersection.
        :return: The list of bounding boxes.
        """
        result = []
        for x, y, z in itertools.product(
            simple_event[SpatialVariables.x.value].simple_sets,
            simple_event[SpatialVariables.y.value].simple_sets,
            simple_event[SpatialVariables.z.value].simple_sets,
        ):
            result.append(
                cls(x.lower, y.lower, z.lower, x.upper, y.upper, z.upper, origin)
            )
        return result

    def intersection_with(self, other: BoundingBox) -> Optional[BoundingBox]:
        """
        Compute the intersection of two bounding boxes.

        :param other: The other bounding box.
        :return: The intersection of the two bounding boxes or None if they do not intersect.
        """
        other_in_same_frame = other.transform_to_origin(self.origin)
        result = self.simple_event.intersection_with(other_in_same_frame.simple_event)
        if result.is_empty():
            return None
        return self.__class__.from_simple_event(result, self.origin)[0]

    def enlarge(
        self,
        min_x: float = 0.0,
        min_y: float = 0,
        min_z: float = 0,
        max_x: float = 0.0,
        max_y: float = 0.0,
        max_z: float = 0.0,
    ):
        """
        Enlarge the axis-aligned bounding box by a given amount in-place.
        :param min_x: The amount to enlarge the minimum x-coordinate
        :param min_y: The amount to enlarge the minimum y-coordinate
        :param min_z: The amount to enlarge the minimum z-coordinate
        :param max_x: The amount to enlarge the maximum x-coordinate
        :param max_y: The amount to enlarge the maximum y-coordinate
        :param max_z: The amount to enlarge the maximum z-coordinate
        """
        self.min_x -= min_x
        self.min_y -= min_y
        self.min_z -= min_z
        self.max_x += max_x
        self.max_y += max_y
        self.max_z += max_z

    def enlarge_all(self, amount: float):
        """
        Enlarge the axis-aligned bounding box in all dimensions by a given amount in-place.

        :param amount: The amount to enlarge the bounding box
        """
        self.enlarge(amount, amount, amount, amount, amount, amount)

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        origin: HomogeneousTransformationMatrix,
    ) -> Self:
        """
        Create a bounding box from a trimesh object.
        :param mesh: The trimesh object.
        :param origin: The origin of the bounding box.
        :return: The bounding box.
        """

        bounds = mesh.bounds
        return cls(
            bounds[0][0],
            bounds[0][1],
            bounds[0][2],
            bounds[1][0],
            bounds[1][1],
            bounds[1][2],
            origin=origin,
        )

    def get_points(self) -> List[Point3]:
        """
        Get the 8 corners of the bounding box as Point3 objects.

        :return: A list of Point3 objects representing the corners of the bounding box.
        """
        return [
            Point3(x, y, z)
            for x in (self.min_x, self.max_x)
            for y in (self.min_y, self.max_y)
            for z in (self.min_z, self.max_z)
        ]

    @classmethod
    def from_min_max(
        cls,
        min_point: Point3,
        max_point: Point3,
        origin: HomogeneousTransformationMatrix,
    ) -> Self:
        """
        Set the axis-aligned bounding box from a minimum and maximum point.

        :param min_point: The minimum point
        :param max_point: The maximum point
        """
        assert (
            min_point.reference_frame == max_point.reference_frame
        ), "The reference frames of the minimum and maximum points must be the same."
        return cls(*min_point.to_np()[:3], *max_point.to_np()[:3], origin=origin)

    def as_shape(self) -> Box:
        scale = Scale(
            x=self.max_x - self.min_x,
            y=self.max_y - self.min_y,
            z=self.max_z - self.min_z,
        )
        x = (self.max_x + self.min_x) / 2 + float(self.origin.x)
        y = (self.max_y + self.min_y) / 2 + float(self.origin.y)
        z = (self.max_z + self.min_z) / 2 + float(self.origin.z)
        origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, z, 0, 0, 0, self.origin.reference_frame
        )
        return Box(origin=origin, scale=scale)

    def transform_to_origin(
        self, reference_T_new_origin: HomogeneousTransformationMatrix
    ) -> Self:
        """
        Transform the bounding box to a different reference frame.
        """
        reference_T_new_origin = HomogeneousTransformationMatrix(
            data=reference_T_new_origin.to_np(),
            reference_frame=reference_T_new_origin.reference_frame,
        )

        new_origin_reference_T_self = self.origin.reference_frame._world.transform(
            self.origin, reference_T_new_origin.reference_frame
        )

        self_T_new_pose = reference_T_new_origin.inverse() @ new_origin_reference_T_self

        # Get all 8 corners of the BB in link-local space
        list_self_T_corner = [
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                self_T_corner
            ).to_np()
            for self_T_corner in self.get_points()
        ]  # shape (8, 3)

        list_reference_T_corner = [
            self_T_new_pose.to_np() @ self_T_corner
            for self_T_corner in list_self_T_corner
        ]

        list_reference_P_corner = [
            reference_T_corner[:3, 3:] for reference_T_corner in list_reference_T_corner
        ]

        # Compute new corner points
        min_corner = np.min(list_reference_P_corner, axis=0)
        max_corner = np.max(list_reference_P_corner, axis=0)

        world_bb = BoundingBox.from_min_max(
            Point3.from_iterable(
                min_corner, reference_frame=reference_T_new_origin.reference_frame
            ),
            Point3.from_iterable(
                max_corner, reference_frame=reference_T_new_origin.reference_frame
            ),
            reference_T_new_origin,
        )

        return world_bb

    def __eq__(self, other: BoundingBox) -> bool:
        return (
            np.isclose(self.min_x, other.min_x)
            and np.isclose(self.min_y, other.min_y)
            and np.isclose(self.min_z, other.min_z)
            and np.isclose(self.max_x, other.max_x)
            and np.isclose(self.max_y, other.max_y)
            and np.isclose(self.max_z, other.max_z)
            and np.allclose(self.origin.to_np(), other.origin.to_np())
        )
