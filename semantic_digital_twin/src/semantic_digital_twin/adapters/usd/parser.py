"""
Reading USD stages into worlds.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import NoneType

import numpy as np
import trimesh
from numpy.typing import NDArray
from typing_extensions import Self, Sequence

from krrood.class_diagrams.mocking import MockedClass, MockedModule

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.usd.exceptions import (
    ScaledArticulatedUSDComponentError,
    UnreadUSDPhysicsJointsError,
    UnsupportedUSDGeometryTypeError,
    UnsupportedUSDPhysicsJointTypeError,
    USDJointChildIsComponentError,
    USDJointLinkOutsideComponentError,
    USDPhysicsJointMissingChildBodyError,
)
from semantic_digital_twin.world_source import WorldModelParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.usd_semantics import USDSemanticLabels
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.specifications.connections import (
    ConnectionSpecification,
    FixedConnectionSpecification,
    ConnectionSpecificationWithEndpoints,
    ScrewAxisConnectionSpecification,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
    ScrewAxis,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
    PrincipalAxes,
    PrincipalMoments,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

# %% pxr fallbacks
#
# Usd.Prim, Usd.Stage, and Gf.Matrix4d are used below as dataclass field types.
# Leaving Usd/Gf unbound when usd-core is missing crashes any code that resolves
# this module's type hints (e.g. ORM/class-diagram generation), not just code that
# actually parses a USD stage. Binding them to these mocks keeps every annotation
# resolvable; actually parsing a stage still fails loudly, via
# krrood.class_diagrams.mocking.MockedClass.


@dataclass
class _MockedUSDPrim(MockedClass):
    """
    Mocked class for Usd.Prim in pxr.
    """


@dataclass
class _MockedUSDStage(MockedClass):
    """
    Mocked class for Usd.Stage in pxr.
    """


@dataclass
class _MockedUSDModule(MockedModule):
    """
    Mocked module for pxr.Usd.
    """

    Prim: type[_MockedUSDPrim] = _MockedUSDPrim
    Stage: type[_MockedUSDStage] = _MockedUSDStage


@dataclass
class _MockedGfMatrix4d(MockedClass):
    """
    Mocked class for Gf.Matrix4d in pxr.
    """


@dataclass
class _MockedGfModule(MockedModule):
    """
    Mocked module for pxr.Gf.
    """

    Matrix4d: type[_MockedGfMatrix4d] = _MockedGfMatrix4d


try:
    from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    logger.warning(
        "usd-core is required for USD parsing. Please install it using "
        "'pip install usd-core'"
    )
    Usd = _MockedUSDModule()
    Gf = _MockedGfModule()

try:
    from pxr import UsdSemantics
except ImportError:
    # UsdSemantics (UsdSemantics.LabelsAPI) is only available from usd-core 24.11
    # onward; an older install simply never yields USDSemanticLabels annotations.
    UsdSemantics = NoneType


def _usd_pose_to_transform(
    position: Gf.Vec3d, rotation: Gf.Quatf, **kwargs
) -> HomogeneousTransformationMatrix:
    """
    Build a transform from a USD position and quaternion rotation.

    :param position: The translation.
    :param rotation: The rotation.
    :param kwargs: Forwarded to ``HomogeneousTransformationMatrix.from_xyz_quaternion``
        (typically ``reference_frame``/``child_frame``).
    :return: The built transform.
    """
    imaginary = rotation.GetImaginary()
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=position[0],
        pos_y=position[1],
        pos_z=position[2],
        quat_x=imaginary[0],
        quat_y=imaginary[1],
        quat_z=imaginary[2],
        quat_w=rotation.GetReal(),
        **kwargs,
    )


def _decompose_local_transform(
    prim: Usd.Prim, link_to_world: Gf.Matrix4d
) -> tuple[HomogeneousTransformationMatrix, Gf.Vec3d]:
    """
    Decompose a prim's pose relative to its enclosing link into a rigid transform and a
    scale, regardless of how the prim's ``xformOpOrder`` was authored.

    :param prim: The prim whose pose to decompose.
    :param link_to_world: The enclosing link's local-to-world transform.
    :return: The prim's rigid pose relative to the link (with no ``reference_frame`` set
        - the caller sets it), and its scale relative to the link.
    """
    prim_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    prim_to_link = prim_to_world * link_to_world.GetInverse()
    transform = Gf.Transform(prim_to_link)
    translation = transform.GetTranslation()
    rotation_quat = transform.GetRotation().GetQuat()
    origin = _usd_pose_to_transform(translation, rotation_quat)
    return origin, transform.GetScale()


class USDGeomPrimType(StrEnum):
    """
    The USD prim type names (``UsdGeom.Xxx.Define``'s ``Xxx``) this parser builds a
    Shape for.
    """

    MESH = "Mesh"
    CUBE = "Cube"
    SPHERE = "Sphere"
    CYLINDER = "Cylinder"

    def create_shape(self, prim: Usd.Prim, link_to_world: Gf.Matrix4d) -> Shape:
        """
        Creates the Shape a prim of this type describes.

        :param prim: The prim to create a shape for.
        :param link_to_world: The enclosing link's local-to-world transform.
        :return: The created shape.
        """
        builders: dict[USDGeomPrimType, type[USDShapeBuilder]] = {
            USDGeomPrimType.MESH: USDMeshShapeBuilder,
            USDGeomPrimType.CUBE: USDCubeShapeBuilder,
            USDGeomPrimType.SPHERE: USDSphereShapeBuilder,
            USDGeomPrimType.CYLINDER: USDCylinderShapeBuilder,
        }
        return builders[self](prim, link_to_world).build()


class USDPhysicsJointType(StrEnum):
    """
    The USD physics joint prim type names this parser builds a Connection for.
    """

    FIXED = "PhysicsFixedJoint"
    REVOLUTE = "PhysicsRevoluteJoint"
    PRISMATIC = "PhysicsPrismaticJoint"

    @property
    def connection_specification_type(self) -> type[ConnectionSpecification]:
        """
        :return: The specification of the connection this joint type becomes.
        """
        return {
            USDPhysicsJointType.FIXED: FixedConnectionSpecification,
            USDPhysicsJointType.REVOLUTE: ScrewAxisConnectionSpecification,
            USDPhysicsJointType.PRISMATIC: ScrewAxisConnectionSpecification,
        }[self]

    def create_screw_axis(self, *, direction: Vector3) -> ScrewAxis:
        """
        :param direction: The unit vector this movable joint moves along or about.
        :return: The screw axis of the connection this movable joint type becomes.
        """
        if self is USDPhysicsJointType.PRISMATIC:
            return ScrewAxisConnection.create_prismatic_screw_axis(axis=direction)
        return ScrewAxisConnection.create_revolute_screw_axis(axis=direction)


class USDAxis(StrEnum):
    """
    A USD local-frame axis token, as authored on a joint's ``axis`` attribute or a
    ``UsdGeom.Cylinder``'s.
    """

    X = "X"
    Y = "Y"
    Z = "Z"

    def vector(self, reference_frame: Body) -> Vector3:
        """
        :param reference_frame: The frame the returned vector is expressed in.
        :return: The unit vector this axis token denotes.
        """
        unit_vectors = {
            USDAxis.X: (1.0, 0.0, 0.0),
            USDAxis.Y: (0.0, 1.0, 0.0),
            USDAxis.Z: (0.0, 0.0, 1.0),
        }
        return Vector3(*unit_vectors[self], reference_frame=reference_frame)


@dataclass
class USDShapeBuilder(ABC):
    """
    Builds the :class:`~semantic_digital_twin.world_description.geometry.Shape` one USD
    geometry prim describes, relative to its enclosing link.
    """

    prim: Usd.Prim
    """
    The prim to build a shape for.
    """

    link_to_world: Gf.Matrix4d
    """
    The enclosing link's local-to-world transform.
    """

    @abstractmethod
    def build(self) -> Shape:
        """
        :return: The shape :attr:`prim` describes.
        """


@dataclass
class USDCubeShapeBuilder(USDShapeBuilder):
    """
    Builds the Box shape a ``UsdGeom.Cube`` prim describes.
    """

    def build(self) -> Box:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        side = UsdGeom.Cube(self.prim).GetSizeAttr().Get()
        return Box(
            origin=origin,
            scale=Scale(side * scale[0], side * scale[1], side * scale[2]),
        )


@dataclass
class USDSphereShapeBuilder(USDShapeBuilder):
    """
    Builds the Sphere shape a ``UsdGeom.Sphere`` prim describes.

    A sphere has no per-axis size, so a non-uniform scale is approximated by its average
    factor across the three axes.
    """

    def build(self) -> Sphere:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        radius = UsdGeom.Sphere(self.prim).GetRadiusAttr().Get()
        average_scale = (scale[0] + scale[1] + scale[2]) / 3.0
        return Sphere(origin=origin, radius=radius * average_scale)


@dataclass
class USDCylinderShapeBuilder(USDShapeBuilder):
    """
    Builds the Cylinder shape a ``UsdGeom.Cylinder`` prim describes.

    :class:`~semantic_digital_twin.world_description.geometry.Cylinder` is always
    aligned with its local Z axis, so a cylinder authored along X or Y gets an extra
    rotation folded into its origin to align it.
    """

    def build(self) -> Cylinder:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        usd_cylinder = UsdGeom.Cylinder(self.prim)
        axis = usd_cylinder.GetAxisAttr().Get()
        # scale is in the prim's own local axes, unaffected by the alignment rotation
        # below (which only reorients origin), so the axis also picks out which scale
        # components are the cylinder's height vs. its two radial directions.
        if axis == USDAxis.X:
            alignment = HomogeneousTransformationMatrix.from_xyz_rpy(pitch=math.pi / 2)
            height_scale, radial_scale = scale[0], (scale[1] + scale[2]) / 2.0
        elif axis == USDAxis.Y:
            alignment = HomogeneousTransformationMatrix.from_xyz_rpy(roll=-math.pi / 2)
            height_scale, radial_scale = scale[1], (scale[0] + scale[2]) / 2.0
        else:
            alignment = HomogeneousTransformationMatrix()
            height_scale, radial_scale = scale[2], (scale[0] + scale[1]) / 2.0
        origin = origin @ alignment
        return Cylinder(
            origin=origin,
            width=usd_cylinder.GetRadiusAttr().Get() * 2 * radial_scale,
            height=usd_cylinder.GetHeightAttr().Get() * height_scale,
        )


@dataclass
class USDMeshShapeBuilder(USDShapeBuilder):
    """
    Builds the Mesh shape one USD mesh prim describes, positioned relative to its link.

    Applied directly to the raw vertex positions rather than split into a rotation,
    translation, and :class:`~semantic_digital_twin.world_description.geometry.Scale`
    for the shape's origin, since a mesh's local-to-link transform can carry a non-
    uniform scale or shear: decomposing a general affine transform into
    translation/rotation/scale is ill-posed in the presence of shear, while applying the
    matrix to the points themselves is exact regardless.
    """

    def build(self) -> Mesh:
        mesh_to_world = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        mesh_to_link = mesh_to_world * self.link_to_world.GetInverse()

        mesh_geometry = UsdGeom.Mesh(self.prim)
        local_vertices = np.array(mesh_geometry.GetPointsAttr().Get())
        vertices = self._transform_points(local_vertices, mesh_to_link)
        faces = self._triangulate(
            mesh_geometry.GetFaceVertexCountsAttr().Get(),
            mesh_geometry.GetFaceVertexIndicesAttr().Get(),
        )
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        texture_file_path = self._diffuse_texture_path(self.prim)
        uv_per_point = self._uv_coordinates(self.prim)
        if texture_file_path is None or uv_per_point is None:
            uv = None
        else:
            uv = uv_per_point[faces.reshape(-1)]

        return Mesh.from_trimesh(
            mesh=trimesh_mesh,
            origin=HomogeneousTransformationMatrix(),
            uv=uv,
            texture_file_path=texture_file_path if uv is not None else None,
        )

    @staticmethod
    def _transform_points(
        points: NDArray[np.float64], matrix: Gf.Matrix4d
    ) -> NDArray[np.float64]:
        """
        Applies a USD transform to an array of points.

        :param points: An ``(n, 3)`` array of points in the transform's source frame.
        :param matrix: The transform to apply.
        :return: An ``(n, 3)`` array of the transformed points.
        """
        points_homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        return (points_homogeneous @ np.array(matrix))[:, :3]

    @staticmethod
    def _diffuse_texture_path(mesh_prim: Usd.Prim) -> str | None:
        """
        Resolves the file path of the diffuse texture bound to a mesh prim's material.

        :param mesh_prim: The mesh prim to look up.
        :return: The resolved path to the diffuse texture image, or ``None`` if the prim
            has no bound material, its surface shader has no ``diffuseColor`` input, or
            that input is not connected to a texture (e.g. a flat colour).
        """
        material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        if not material:
            return None

        surface_source = material.GetSurfaceOutput().GetConnectedSource()
        if surface_source is None:
            return None
        surface_shader = UsdShade.Shader(surface_source[0])

        diffuse_input = surface_shader.GetInput("diffuseColor")
        diffuse_source = diffuse_input.GetConnectedSource() if diffuse_input else None
        if diffuse_source is None:
            return None
        texture_shader = UsdShade.Shader(diffuse_source[0])

        file_input = texture_shader.GetInput("file")
        asset_path = file_input.Get() if file_input else None
        if asset_path is None or not asset_path.resolvedPath:
            return None
        return asset_path.resolvedPath

    @staticmethod
    def _uv_coordinates(mesh_prim: Usd.Prim) -> NDArray[np.float64] | None:
        """
        Reads a mesh prim's per-point UV coordinates from its ``st`` primvar.

        :param mesh_prim: The mesh prim to look up.
        :return: An ``(n_points, 2)`` array of UV coordinates, or ``None`` if the prim
            has no ``st`` primvar, or its interpolation is not per-point
            (``vertex``/``varying``).
        """
        primvar = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
        if not primvar.IsDefined():
            return None
        if primvar.GetInterpolation() not in (
            UsdGeom.Tokens.vertex,
            UsdGeom.Tokens.varying,
        ):
            return None
        values = primvar.Get()
        if not values:
            return None
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _triangulate(
        face_vertex_counts: Sequence[int], face_vertex_indices: Sequence[int]
    ) -> NDArray[np.int64]:
        """
        Fan-triangulates a USD mesh's polygonal faces.

        :param face_vertex_counts: The number of vertices of each face.
        :param face_vertex_indices: The faces' vertex indices, flattened in
            ``face_vertex_counts`` order.
        :return: An ``(n, 3)`` array of triangle vertex indices.
        """
        triangles = []
        cursor = 0
        for count in face_vertex_counts:
            face = face_vertex_indices[cursor : cursor + count]
            for i in range(1, count - 1):
                triangles.append((face[0], face[i], face[i + 1]))
            cursor += count
        return np.array(triangles, dtype=np.int64)


@dataclass(kw_only=True)
class USDJointedLinks:
    """
    The link bodies physics joints connect and the connections the joints describe,
    built before any world is modified.
    """

    link_bodies: dict[str, Body] = field(default_factory=dict)
    """
    The link bodies, by the stage path of their root prim.
    """

    connections: list[ConnectionSpecificationWithEndpoints] = field(
        default_factory=list
    )
    """
    The connections the joints describe.
    """


@dataclass(kw_only=True)
class USDStageParser(WorldModelParser, ABC):
    """
    Base for the parsers that turn a Universal Scene Description (USD) stage, read with
    the ``usd-core`` package (``pxr``), into a world. The world keeps the up axis and
    unit scale the stage declares in its metadata.
    """

    stage: Usd.Stage
    """
    The USD stage to parse, composed from all its layers (references, sublayers, ...);
    :meth:`from_file` opens one from a file.
    """

    prefix: str | None = None
    """
    The prefix for every name used in this world.
    """

    path_resolver: PathResolver = field(default_factory=CompositePathResolver)
    """
    The path resolver used for the asset references of this stage.
    """

    def __post_init__(self):
        """
        Defaults :attr:`prefix` to the file name stem of the stage's root layer.
        """
        if self.prefix is None:
            self.prefix = Path(self.stage.GetRootLayer().identifier).stem

    # %% construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
    ) -> Self:
        """
        Creates a parser for a USD stage file.

        :param file_path: The path of the stage file to parse.
        :param prefix: The prefix for every name used in this world.
        :param path_resolver: The resolver for the asset references of the stage.
        :return: A parser for the described world.
        """
        path_resolver = path_resolver or CompositePathResolver()
        resolved_path = path_resolver.resolve(file_path)
        parser = cls(stage=Usd.Stage.Open(resolved_path), prefix=prefix)
        parser.path_resolver = path_resolver
        return parser

    # %% diagnostics

    @property
    def source_description(self) -> str:
        """
        :return: A human-readable identifier of :attr:`stage`, used in error messages -
            its file path if it was opened from one, its in-memory layer identifier
            otherwise.
        """
        return self.stage.GetRootLayer().identifier

    # %% stage structure

    def _root_prim(self) -> Usd.Prim | None:
        """
        :return: The stage's default prim, or its single top-level prim if it has no
            default prim and exactly one top-level prim, or ``None`` if neither
            identifies a single prim unambiguously.
        """
        default_prim = self.stage.GetDefaultPrim()
        if default_prim.IsValid():
            return default_prim

        top_level_prims = self._top_level_prims()
        if len(top_level_prims) != 1:
            return None
        return top_level_prims[0]

    def _create_world_with_root_body(self) -> World:
        """
        :return: A world whose root body is named after :meth:`_root_prim`, or after
            :attr:`prefix` if there is none.
        """
        root_prim = self._root_prim()
        root_body_name = root_prim.GetName() if root_prim is not None else self.prefix
        return World.create_with_root_body(root_body_name, self.prefix)

    def _top_level_prims(self) -> list[Usd.Prim]:
        """
        :return: Every top-level prim of the stage (the pseudo-root's direct children).
        """
        return list(self.stage.GetPseudoRoot().GetChildren())

    def _physics_joint_prims(self) -> list[Usd.Prim]:
        """
        :return: Every :class:`~pxr.UsdPhysics.Joint` prim of the stage, those inside
            instances of instanceable prims included.
        """
        return [
            prim
            for prim in self.stage.Traverse(Usd.TraverseInstanceProxies())
            if prim.IsA(UsdPhysics.Joint)
        ]

    @staticmethod
    def _local_to_world(*, prim: Usd.Prim) -> Gf.Matrix4d:
        """
        :param prim: The prim whose transform to compute.
        :return: The prim's transform relative to the stage's world frame, including
            its scale.
        """
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

    @staticmethod
    def _pose_in_world(*, prim: Usd.Prim) -> Gf.Matrix4d:
        """
        :param prim: The prim whose pose to compute.
        :return: The prim's transform relative to the stage's world frame, without its
            scale.
        """
        local_to_world = Gf.Transform(USDStageParser._local_to_world(prim=prim))
        return Gf.Matrix4d().SetTransform(
            local_to_world.GetRotation(), local_to_world.GetTranslation()
        )

    # %% links and shapes

    def _create_link_body(
        self,
        *,
        link_prim: Usd.Prim,
        name: str,
        link_to_world: Gf.Matrix4d,
        excluded_prim_paths: frozenset[Sdf.Path] = frozenset(),
    ) -> Body:
        """
        Creates the Body for one rigid link, with a Shape for every mesh/primitive in
        its USD subtree and its :class:`~pxr.UsdPhysics.MassAPI` inertial properties, if
        applied.

        :param link_prim: The link's root USD prim.
        :param name: The body's name, prefixed with :attr:`prefix`.
        :param link_to_world: The body frame's transform relative to the stage's world
            frame; any scale it leaves out stays in the shapes.
        :param excluded_prim_paths: The prims below ``link_prim`` whose subtrees are
            other bodies, so their shapes are left out.
        :return: The created body, not yet added to a world.
        """
        shapes = self._shapes_in_subtree(
            link_prim=link_prim,
            link_to_world=link_to_world,
            excluded_prim_paths=excluded_prim_paths,
        )
        shape_collection = ShapeCollection(shapes)
        body = Body(
            name=PrefixedName(name, self.prefix),
            visual=shape_collection,
            collision=shape_collection,
        )
        inertial = self._parse_inertial(link_prim=link_prim, body=body)
        if inertial is not None:
            body.inertial = inertial
        return body

    def _create_free_body(
        self,
        *,
        prim: Usd.Prim,
        name: str,
        excluded_prim_paths: frozenset[Sdf.Path] = frozenset(),
    ) -> Body:
        """
        :param prim: The root prim of an object the stage connects to nothing.
        :param name: The body's name, prefixed with :attr:`prefix`.
        :param excluded_prim_paths: The prims below ``prim`` whose subtrees are other
            bodies, so their shapes are left out.
        :return: The object's body, whose frame is the prim's pose without its scale,
            not yet added to a world.
        """
        return self._create_link_body(
            link_prim=prim,
            name=name,
            link_to_world=self._pose_in_world(prim=prim),
            excluded_prim_paths=excluded_prim_paths,
        )

    def _add_free_body(self, *, world: World, prim: Usd.Prim, body: Body) -> None:
        """
        Adds a body from :meth:`_create_free_body` to ``world``, attached to its root by a
        :class:`~semantic_digital_twin.world_description.connections.Connection6DoF` at
        the prim's pose. Must be called inside ``world``'s modification context.

        :param world: The world, whose root body's frame is the stage's world frame.
        :param prim: The prim ``body`` was created from.
        :param body: The body to add.
        """
        world.add_body(body)
        self._attach_semantic_labels(world=world, prim=prim, body=body)
        connection = Connection6DoF.create_with_dofs(
            world=world, parent=world.root, child=body
        )
        world.add_connection(connection)
        # Gf matrices act on row vectors, so the transpose is the pose.
        connection.origin = HomogeneousTransformationMatrix(
            data=np.array(self._pose_in_world(prim=prim)).T,
            reference_frame=world.root,
        )

    def _shapes_in_subtree(
        self,
        *,
        link_prim: Usd.Prim,
        link_to_world: Gf.Matrix4d,
        excluded_prim_paths: frozenset[Sdf.Path] = frozenset(),
    ) -> list[Shape]:
        """
        Creates the Shape for every mesh/primitive prim in a link's subtree, inside
        instances of instanceable prims too.

        :param link_prim: The root of the subtree to search for shapes.
        :param link_to_world: The transform, relative to the stage's world frame, of
            the frame the shapes are positioned relative to.
        :param excluded_prim_paths: The prims whose subtrees are left out.
        :return: The created shapes.
        """
        shapes = []
        prim_range = iter(Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()))
        for prim in prim_range:
            if prim.GetPath() in excluded_prim_paths:
                prim_range.PruneChildren()
                continue
            shape = self._create_shape(prim=prim, link_to_world=link_to_world)
            if shape is not None:
                shapes.append(shape)
        return shapes

    def _create_shape(
        self, *, prim: Usd.Prim, link_to_world: Gf.Matrix4d
    ) -> Shape | None:
        """
        Creates the Shape a mesh/primitive prim describes. A prim that is no renderable
        geometric primitive (an ``Xform``, a camera, a shader, ...) is skipped.

        :param prim: The prim to create a shape for.
        :param link_to_world: The enclosing link's local-to-world transform.
        :return: The created shape, or ``None`` if ``prim`` is not shape geometry.
        :raises UnsupportedUSDGeometryTypeError: If ``prim`` is a renderable geometric
            primitive of a type this parser does not build a Shape for.
        """
        type_name = prim.GetTypeName()
        try:
            geom_type = USDGeomPrimType(type_name)
        except ValueError as error:
            if prim.IsA(UsdGeom.Gprim):
                raise UnsupportedUSDGeometryTypeError(
                    file_path=self.source_description,
                    prim_path=str(prim.GetPath()),
                    geometry_type=type_name,
                    supported_types=list(USDGeomPrimType),
                ) from error
            return None
        return geom_type.create_shape(prim, link_to_world)

    # %% joints

    def _describe_joint(
        self,
        *,
        joint_prim: Usd.Prim,
        root_body: Body,
        link_bodies: dict[str, Body],
    ) -> ConnectionSpecificationWithEndpoints:
        """
        Validates and specifies the joint one physics joint prim states, without
        touching a world - a World left partway through a failed modification is
        unusable, so every joint is described before any world modification begins.

        :param joint_prim: The USD physics joint prim (a Fixed/Revolute/PrismaticJoint).
        :param root_body: The world's root body, used as the connection's parent if the
            joint's ``body0`` relationship has no target.
        :param link_bodies: The link Bodies created so far, by stage path - extended in
            place as new links are resolved.
        :return: The specification of the joint.
        :raises UnsupportedUSDPhysicsJointTypeError: If the joint's type has no
            Connection counterpart.
        :raises USDPhysicsJointMissingChildBodyError: If the joint's ``body1``
            relationship has no target.
        """
        try:
            usd_joint_type = USDPhysicsJointType(joint_prim.GetTypeName())
        except ValueError as error:
            raise UnsupportedUSDPhysicsJointTypeError(
                file_path=self.source_description,
                joint_path=str(joint_prim.GetPath()),
                joint_type=joint_prim.GetTypeName(),
                supported_types=list(USDPhysicsJointType),
            ) from error
        specification_type = usd_joint_type.connection_specification_type

        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body1_targets:
            raise USDPhysicsJointMissingChildBodyError(
                file_path=self.source_description, joint_path=str(joint_prim.GetPath())
            )
        parent = (
            self._resolve_link_body(link_bodies=link_bodies, prim_path=body0_targets[0])
            if body0_targets
            else root_body
        )
        child = self._resolve_link_body(
            link_bodies=link_bodies, prim_path=body1_targets[0]
        )

        parent_T_connection = _usd_pose_to_transform(
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
            reference_frame=parent,
        )
        # UsdPhysics.Joint documents localPos1/localRot1 as the joint frame's pose
        # relative to body1 (child_T_connection), the opposite of what is needed here -
        # unlike localPos0/localRot0, which is already parent_T_connection as authored.
        child_T_connection = _usd_pose_to_transform(
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
            reference_frame=child,
        )
        connection_T_child = child_T_connection.inverse()

        if specification_type is FixedConnectionSpecification:
            return ConnectionSpecificationWithEndpoints(
                connection_specification=FixedConnectionSpecification(
                    connection_T_child=connection_T_child
                ),
                parent=parent,
                child=child,
                parent_T_connection=parent_T_connection,
            )

        axis_joint = (
            UsdPhysics.RevoluteJoint(joint_prim)
            if usd_joint_type is USDPhysicsJointType.REVOLUTE
            else UsdPhysics.PrismaticJoint(joint_prim)
        )
        axis = USDAxis(axis_joint.GetAxisAttr().Get()).vector(reference_frame=parent)
        lower = axis_joint.GetLowerLimitAttr().Get()
        upper = axis_joint.GetUpperLimitAttr().Get()
        if usd_joint_type is USDPhysicsJointType.REVOLUTE:
            lower, upper = math.radians(lower), math.radians(upper)
        if lower > upper:
            # Seen authored this way on a mirrored part (e.g. one blade of a pair of
            # scissors): the pair's joints share one axis convention, so the mirrored
            # joint's authored "lower"/"upper" swap relative to it even though both
            # describe the same-sized range of motion. The DOF's own lower/upper are
            # just its two extremes, so swapping the values (not negating them) keeps
            # the authored range of motion intact.
            lower, upper = upper, lower
        return ConnectionSpecificationWithEndpoints(
            connection_specification=specification_type(
                connection_T_child=connection_T_child,
                screw_axis=usd_joint_type.create_screw_axis(direction=axis),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=lower, upper_position=upper
                ),
            ),
            parent=parent,
            child=child,
            parent_T_connection=parent_T_connection,
        )

    def _resolve_link_body(
        self, *, link_bodies: dict[str, Body], prim_path: Sdf.Path
    ) -> Body:
        """
        Resolves a USD prim path to its link Body, creating (and caching in
        ``link_bodies``) the Body on first use.

        :param link_bodies: The link Bodies created so far, by stage path - extended in
            place if ``prim_path`` has not been resolved yet.
        :param prim_path: The stage path a joint's ``body0``/``body1`` relationship
            targets.
        :return: The resolved link Body.
        """
        prim = self._link_prim(prim_path=prim_path)
        path_string = str(prim.GetPath())
        if path_string not in link_bodies:
            link_bodies[path_string] = self._create_joint_link_body(prim=prim)
        return link_bodies[path_string]

    def _link_prim(self, *, prim_path: Sdf.Path) -> Usd.Prim:
        """
        :param prim_path: The stage path a joint's ``body0``/``body1`` relationship
            targets.
        :return: The root prim of the link the path names.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.GetTypeName() == USDGeomPrimType.MESH:
            # Some joints target a link's mesh prim instead of its enclosing Xform;
            # both resolve to the Xform, so a link is never split into two bodies.
            return prim.GetParent()
        return prim

    def _add_jointed_links(
        self, *, world: World, jointed_links: USDJointedLinks
    ) -> None:
        """
        Adds the link bodies of ``jointed_links``, with their semantic labels, and
        their connections to ``world``. Must be called inside ``world``'s modification
        context.

        :param world: The world to add to.
        :param jointed_links: The links and connections to add.
        """
        for path_string, link_body in jointed_links.link_bodies.items():
            world.add_body(link_body)
            self._attach_semantic_labels(
                world=world, prim=self.stage.GetPrimAtPath(path_string), body=link_body
            )
        for connection in jointed_links.connections:
            connection.connect(world)

    @abstractmethod
    def _create_joint_link_body(self, *, prim: Usd.Prim) -> Body:
        """
        :param prim: The root prim of a link a physics joint connects.
        :return: The link's body, not yet added to a world.
        """

    # %% inertials

    @staticmethod
    def _parse_inertial(*, link_prim: Usd.Prim, body: Body) -> Inertial | None:
        """
        Parses a link prim's :class:`~pxr.UsdPhysics.MassAPI` inertial properties.

        :param link_prim: The link's root USD prim.
        :param body: The body the properties belong to, used as their reference frame.
        :return: The inertial properties, or ``None`` if ``UsdPhysics.MassAPI`` is not
            applied to the prim.
        """
        if not link_prim.HasAPI(UsdPhysics.MassAPI):
            return None

        mass_api = UsdPhysics.MassAPI(link_prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None or mass <= 0.0:
            return None

        center_of_mass = mass_api.GetCenterOfMassAttr().Get() or (0.0, 0.0, 0.0)
        diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0)
        principal_axes = mass_api.GetPrincipalAxesAttr().Get()

        principal_moments = PrincipalMoments.from_values(
            i1=diagonal_inertia[0], i2=diagonal_inertia[1], i3=diagonal_inertia[2]
        )
        axes_rotation = _usd_pose_to_transform(
            Gf.Vec3d(0, 0, 0),
            principal_axes if principal_axes is not None else Gf.Quatf(1, 0, 0, 0),
        )
        inertia_tensor = InertiaTensor.from_principal_moments_and_axes(
            moments=principal_moments,
            axes=PrincipalAxes.from_rotation_matrix(
                RotationMatrix(data=axes_rotation.to_np()[:3, :3])
            ),
        )

        return Inertial(
            mass=mass,
            center_of_mass=Point3(*center_of_mass, reference_frame=body),
            inertia=inertia_tensor,
        )

    # %% semantics

    def _attach_root_semantic_labels(self, *, world: World) -> None:
        """
        Attaches the semantic labels of :meth:`_root_prim`, if there is one, to the root
        body of ``world``. Must be called inside ``world``'s modification context.

        :param world: The world whose root body was created for :meth:`_root_prim`.
        """
        root_prim = self._root_prim()
        if root_prim is None:
            return
        self._attach_semantic_labels(world=world, prim=root_prim, body=world.root)

    @staticmethod
    def _attach_semantic_labels(*, world: World, prim: Usd.Prim, body: Body) -> None:
        """
        Attaches a :class:`USDSemanticLabels` annotation to ``body`` for every
        ``UsdSemantics.LabelsAPI`` taxonomy directly authored on ``prim``, if any.

        Must be called inside ``world``'s modification context, alongside adding
        ``body`` itself.

        :param world: The world to add the annotation to.
        :param prim: The USD prim ``body`` was built from.
        :param body: The already-added body the annotation's root is.
        """
        for taxonomy, labels in USDStageParser._read_semantic_labels(prim=prim).items():
            world.add_semantic_annotation(
                USDSemanticLabels(root=body, taxonomy=taxonomy, labels=labels)
            )

    @staticmethod
    def _read_semantic_labels(*, prim: Usd.Prim) -> dict[str, list[str]]:
        """
        Reads every ``UsdSemantics.LabelsAPI`` taxonomy directly authored on a prim.

        Only labels authored directly on ``prim`` are read, the same way
        :meth:`_parse_inertial` only reads a link's own ``UsdPhysics.MassAPI`` - not
        the taxonomies USD's inheritance semantics would additionally consider
        accumulated from an ancestor prim.

        :param prim: The prim to read semantic labels from.
        :return: A mapping of taxonomy to the labels authored under it, empty if
            ``usd-core`` predates ``UsdSemantics`` or the prim has none.
        """
        if UsdSemantics is NoneType:
            return {}
        return {
            taxonomy: list(
                UsdSemantics.LabelsAPI.Get(prim, taxonomy).GetLabelsAttr().Get() or ()
            )
            for taxonomy in UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim)
        }


@dataclass(kw_only=True)
class USDParser(USDStageParser):
    """
    Parses a USD stage of one asset into a world: its rigid links are the prims its
    :class:`~pxr.UsdPhysics.Joint` prims connect, not its ``Xform`` hierarchy, and each
    link must be the ``body1`` of exactly one joint. A stage without physics joints
    becomes one body holding every shape under its root.
    """

    # %% entry point

    def create_world(self) -> World:
        """
        Differs from
        :meth:`~semantic_digital_twin.world_source.WorldSource.create_world` in the
        joints it rejects.

        :raises UnsupportedUSDPhysicsJointTypeError: If the stage contains a physics
            joint of a type this parser does not build a Connection for.
        :raises USDPhysicsJointMissingChildBodyError: If a physics joint's ``body1``
            relationship has no target.
        """
        joint_prims = self._physics_joint_prims()
        if not joint_prims:
            return self._parse_jointless_stage()
        return self._parse_joint_graph(joint_prims)

    def _parse_joint_graph(self, joint_prims: list[Usd.Prim]) -> World:
        """
        Builds the world for a stage with physics joints.

        The joint graph's own root (a joint's unset ``body0``, the USD convention for
        "the stage's own frame") becomes a Body named after :meth:`_root_prim`, or a
        synthetic one if that is ambiguous - unlike a joint's ``body1``, which always
        names a specific link, an unset ``body0`` never depended on the stage having an
        identifiable root prim in the first place, so there is nothing to name it after.

        :param joint_prims: Every ``UsdPhysics.Joint`` prim in the stage.
        :return: The parsed world.
        """
        world = self._create_world_with_root_body()
        link_bodies: dict[str, Body] = {}
        connections = [
            self._describe_joint(
                joint_prim=joint_prim, root_body=world.root, link_bodies=link_bodies
            )
            for joint_prim in joint_prims
        ]
        jointed_links = USDJointedLinks(
            link_bodies=link_bodies, connections=connections
        )
        with world.modify_world():
            self._attach_root_semantic_labels(world=world)
            self._add_jointed_links(world=world, jointed_links=jointed_links)
        return world

    def _parse_jointless_stage(self) -> World:
        """
        Builds the world for a stage with no physics joints at all: a single body for
        its root prim (see :meth:`_root_prim`), or, without one, a synthetic root with
        every top-level prim attached at its pose as its own free body.

        :return: The parsed world.
        """
        world = self._create_world_with_root_body()
        root_prim = self._root_prim()
        if root_prim is not None:
            shapes = self._shapes_in_subtree(
                link_prim=root_prim, link_to_world=self._local_to_world(prim=root_prim)
            )
            shape_collection = ShapeCollection(shapes, reference_frame=world.root)
            shape_collection.transform_all_shapes_to_own_frame()
            world.root.visual = shape_collection
            world.root.collision = shape_collection
            with world.modify_world():
                self._attach_root_semantic_labels(world=world)
            return world

        top_level_prims = self._top_level_prims()
        bodies = [
            self._create_free_body(prim=prim, name=prim.GetName())
            for prim in top_level_prims
        ]
        with world.modify_world():
            for prim, body in zip(top_level_prims, bodies):
                self._add_free_body(world=world, prim=prim, body=body)
        return world

    def _create_joint_link_body(self, *, prim: Usd.Prim) -> Body:
        return self._create_link_body(
            link_prim=prim,
            name=prim.GetName(),
            link_to_world=self._local_to_world(prim=prim),
        )


@dataclass(kw_only=True)
class USDSceneContent:
    """
    What a USD scene holds below its root prim, split at its components.
    """

    component_prims: list[Usd.Prim] = field(default_factory=list)
    """
    Every prim of kind ``component`` below the root, none of them inside another.
    """

    root_shapes: list[Shape] = field(default_factory=list)
    """
    The shapes of the geometry outside every component, positioned relative to the
    stage's world frame.
    """


@dataclass(kw_only=True)
class USDComponent:
    """
    A component of a USD scene, with the physics joints inside it and its body.
    """

    prim: Usd.Prim
    """
    The component's prim.
    """

    joint_prims: list[Usd.Prim]
    """
    The physics joint prims inside the component.
    """

    body: Body
    """
    The component's body, without the geometry of the links its joints connect.
    """


@dataclass(kw_only=True)
class USDComponentParser(USDStageParser):
    """
    Parses a USD scene into one body per prim of kind ``component`` (a placed asset
    such as a chair), named after its path below the root prim and attached to the
    root as authored, even where the model hierarchy above it is invalid. The physics
    joints inside a component connect its links to it, as :class:`USDParser` connects
    the links of an asset to its root. Geometry outside every component belongs to the
    root body.
    """

    def create_world(self) -> World:
        """
        Differs from
        :meth:`~semantic_digital_twin.world_source.WorldSource.create_world` in the
        stages it rejects.

        :raises UnreadUSDPhysicsJointsError: If the stage contains physics joints outside
            every component.
        :raises USDJointChildIsComponentError: If a physics joint's ``body1`` is its own
            component.
        :raises USDJointLinkOutsideComponentError: If a physics joint inside a
            component connects a link outside it.
        :raises ScaledArticulatedUSDComponentError: If a link a physics joint inside a
            component connects is scaled.
        :raises UnsupportedUSDPhysicsJointTypeError: If a component contains a physics
            joint of a type this parser does not build a Connection for.
        :raises USDPhysicsJointMissingChildBodyError: If a physics joint's ``body1``
            relationship has no target.
        """
        world = self._create_world_with_root_body()
        # Every shape and joint is built before the world is modified: a World left
        # partway through a failed modification is unusable.
        content = self._scene_content(scene_root=self._scene_root())
        root_shapes = ShapeCollection(content.root_shapes, reference_frame=world.root)
        world.root.visual = root_shapes
        world.root.collision = root_shapes
        components = self._create_components(component_prims=content.component_prims)
        jointed_links = self._describe_component_joints(components=components)
        with world.modify_world():
            self._attach_root_semantic_labels(world=world)
            for component in components:
                self._add_free_body(
                    world=world, prim=component.prim, body=component.body
                )
            self._add_jointed_links(world=world, jointed_links=jointed_links)
        return world

    def _create_components(
        self, *, component_prims: list[Usd.Prim]
    ) -> list[USDComponent]:
        """
        :param component_prims: The components of the scene, none inside another.
        :return: Each component with its joints and its body.
        :raises UnreadUSDPhysicsJointsError: If a physics joint is inside no component.
        """
        joints_by_component = self._joints_by_component(component_prims=component_prims)
        components = []
        for component_prim in component_prims:
            joint_prims = joints_by_component[component_prim.GetPath()]
            body = self._create_free_body(
                prim=component_prim,
                name=self._name_below_scene_root(prim=component_prim),
                excluded_prim_paths=self._linked_prim_paths(joint_prims=joint_prims)
                - {component_prim.GetPath()},
            )
            components.append(
                USDComponent(prim=component_prim, joint_prims=joint_prims, body=body)
            )
        return components

    def _describe_component_joints(
        self, *, components: list[USDComponent]
    ) -> USDJointedLinks:
        """
        :param components: The components of the scene.
        :return: The links the physics joints inside the components connect and the
            connections the joints describe. A joint attached to its component, or to
            nothing, attaches its link to the component's body.
        :raises USDJointChildIsComponentError: If a joint's ``body1`` is its component.
        :raises USDJointLinkOutsideComponentError: If a joint connects a link outside its
            component.
        :raises ScaledArticulatedUSDComponentError: If a link a joint connects is scaled.
        """
        component_paths = {str(component.prim.GetPath()) for component in components}
        # Holds the components too, so a joint's body0 resolves to its component's body.
        bodies_by_path = {
            str(component.prim.GetPath()): component.body for component in components
        }
        connections = []
        for component in components:
            for joint_prim in component.joint_prims:
                self._validate_joint_inside_component(
                    component=component, joint_prim=joint_prim
                )
                connections.append(
                    self._describe_joint(
                        joint_prim=joint_prim,
                        root_body=component.body,
                        link_bodies=bodies_by_path,
                    )
                )
        return USDJointedLinks(
            link_bodies={
                path_string: body
                for path_string, body in bodies_by_path.items()
                if path_string not in component_paths
            },
            connections=connections,
        )

    def _create_joint_link_body(self, *, prim: Usd.Prim) -> Body:
        return self._create_free_body(
            prim=prim, name=self._name_below_scene_root(prim=prim)
        )

    def _scene_root(self) -> Usd.Prim:
        """
        :return: The prim whose subtree is the scene: the root prim, or the stage's
            pseudo-root if there is none.
        """
        root_prim = self._root_prim()
        return root_prim if root_prim is not None else self.stage.GetPseudoRoot()

    def _name_below_scene_root(self, *, prim: Usd.Prim) -> str:
        """
        :param prim: A prim of the scene.
        :return: The prim's path below :meth:`_scene_root`, which tells apart the many
            instances a scene places of one asset.
        """
        return str(prim.GetPath().MakeRelativePath(self._scene_root().GetPath()))

    def _joints_by_component(
        self, *, component_prims: list[Usd.Prim]
    ) -> dict[Sdf.Path, list[Usd.Prim]]:
        """
        :param component_prims: The components of the scene, none inside another.
        :return: The physics joint prims inside each component, by its path.
        :raises UnreadUSDPhysicsJointsError: If a physics joint is inside no component.
        """
        joints_by_component = {
            component_prim.GetPath(): [] for component_prim in component_prims
        }
        unread_joint_paths = []
        for joint_prim in self._physics_joint_prims():
            joint_path = joint_prim.GetPath()
            component_paths = [
                component_path
                for component_path in joints_by_component
                if joint_path.HasPrefix(component_path)
            ]
            if not component_paths:
                unread_joint_paths.append(str(joint_path))
                continue
            joints_by_component[component_paths[0]].append(joint_prim)
        if unread_joint_paths:
            raise UnreadUSDPhysicsJointsError(
                file_path=self.source_description, joint_paths=unread_joint_paths
            )
        return joints_by_component

    def _validate_joint_inside_component(
        self, *, component: USDComponent, joint_prim: Usd.Prim
    ) -> None:
        """
        :param component: A component of the scene.
        :param joint_prim: A physics joint prim inside the component.
        :raises USDJointChildIsComponentError: If the joint's ``body1`` is the
            component.
        :raises USDJointLinkOutsideComponentError: If the joint connects a link outside
            the component.
        :raises ScaledArticulatedUSDComponentError: If a link the joint connects is
            scaled; an unset ``body0`` is the component.
        """
        joint = UsdPhysics.Joint(joint_prim)
        component_path = component.prim.GetPath()
        child_paths = self._link_paths(relationship=joint.GetBody1Rel())
        if component_path in child_paths:
            raise USDJointChildIsComponentError(
                file_path=self.source_description,
                joint_path=str(joint_prim.GetPath()),
                component_path=str(component_path),
            )
        parent_paths = self._link_paths(relationship=joint.GetBody0Rel()) or [
            component_path
        ]
        for link_path in parent_paths + child_paths:
            if not link_path.HasPrefix(component_path):
                raise USDJointLinkOutsideComponentError(
                    file_path=self.source_description,
                    joint_path=str(joint_prim.GetPath()),
                    component_path=str(component_path),
                    link_path=str(link_path),
                )
            link_to_world = self._local_to_world(
                prim=self.stage.GetPrimAtPath(link_path)
            )
            if not np.allclose(Gf.Transform(link_to_world).GetScale(), 1.0):
                raise ScaledArticulatedUSDComponentError(
                    file_path=self.source_description,
                    joint_path=str(joint_prim.GetPath()),
                    scaled_prim_path=str(link_path),
                )

    def _link_paths(self, *, relationship: Usd.Relationship) -> list[Sdf.Path]:
        """
        :param relationship: A physics joint's ``body0`` or ``body1`` relationship.
        :return: The paths of the root prims of the links it targets.
        """
        return [
            self._link_prim(prim_path=target).GetPath()
            for target in relationship.GetTargets()
        ]

    def _linked_prim_paths(self, *, joint_prims: list[Usd.Prim]) -> frozenset[Sdf.Path]:
        """
        :param joint_prims: Physics joint prims.
        :return: The paths of the root prims of the links the joints connect.
        """
        return frozenset(
            link_path
            for joint_prim in joint_prims
            for relationship in (
                UsdPhysics.Joint(joint_prim).GetBody0Rel(),
                UsdPhysics.Joint(joint_prim).GetBody1Rel(),
            )
            for link_path in self._link_paths(relationship=relationship)
        )

    def _scene_content(self, *, scene_root: Usd.Prim) -> USDSceneContent:
        """
        :param scene_root: The prim whose subtree is the scene.
        :return: The components below ``scene_root``, inside instances of instanceable
            prims too, and the shapes outside them.
        """
        content = USDSceneContent()
        world_frame = Gf.Matrix4d(1.0)
        prim_range = iter(Usd.PrimRange(scene_root, Usd.TraverseInstanceProxies()))
        for prim in prim_range:
            is_component = Usd.ModelAPI(prim).IsKind(
                Kind.Tokens.component, Usd.ModelAPI.KindValidationNone
            )
            if prim != scene_root and is_component:
                content.component_prims.append(prim)
                prim_range.PruneChildren()
                continue
            shape = self._create_shape(prim=prim, link_to_world=world_frame)
            if shape is not None:
                content.root_shapes.append(shape)
        return content
