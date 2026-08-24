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
from typing_extensions import Dict, List, Optional, Sequence, Tuple, Type

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdGeometryTypeError,
    UnsupportedUsdPhysicsJointTypeError,
    UsdPhysicsJointMissingChildBodyError,
)
from semantic_digital_twin.adapters.world_model_parser import (
    JointDescription,
    WorldModelParser,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection,
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
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

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    logger.warning(
        "usd-core is required for USD parsing. Please install it using "
        "'pip install usd-core'"
    )

try:
    from pxr import UsdSemantics
except ImportError:
    # UsdSemantics (UsdSemantics.LabelsAPI) is only available from usd-core 24.11
    # onward; an older install simply never yields UsdSemanticLabels annotations.
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
) -> Tuple[HomogeneousTransformationMatrix, Gf.Vec3d]:
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


class UsdGeomPrimType(StrEnum):
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
        builders: Dict[UsdGeomPrimType, Type[UsdShapeBuilder]] = {
            UsdGeomPrimType.MESH: UsdMeshShapeBuilder,
            UsdGeomPrimType.CUBE: UsdCubeShapeBuilder,
            UsdGeomPrimType.SPHERE: UsdSphereShapeBuilder,
            UsdGeomPrimType.CYLINDER: UsdCylinderShapeBuilder,
        }
        return builders[self](prim, link_to_world).build()


class UsdPhysicsJointType(StrEnum):
    """
    The USD physics joint prim type names this parser builds a Connection for.
    """

    FIXED = "PhysicsFixedJoint"
    REVOLUTE = "PhysicsRevoluteJoint"
    PRISMATIC = "PhysicsPrismaticJoint"

    @property
    def connection_type(self) -> Type[Connection]:
        """
        :return: The Connection class this joint type becomes.
        """
        return {
            UsdPhysicsJointType.FIXED: FixedConnection,
            UsdPhysicsJointType.REVOLUTE: RevoluteConnection,
            UsdPhysicsJointType.PRISMATIC: PrismaticConnection,
        }[self]


class UsdAxis(StrEnum):
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
            UsdAxis.X: (1.0, 0.0, 0.0),
            UsdAxis.Y: (0.0, 1.0, 0.0),
            UsdAxis.Z: (0.0, 0.0, 1.0),
        }
        return Vector3(*unit_vectors[self], reference_frame=reference_frame)


@dataclass
class UsdShapeBuilder(ABC):
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
class UsdCubeShapeBuilder(UsdShapeBuilder):
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
class UsdSphereShapeBuilder(UsdShapeBuilder):
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
class UsdCylinderShapeBuilder(UsdShapeBuilder):
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
        if axis == UsdAxis.X:
            alignment = HomogeneousTransformationMatrix.from_xyz_rpy(pitch=math.pi / 2)
            height_scale, radial_scale = scale[0], (scale[1] + scale[2]) / 2.0
        elif axis == UsdAxis.Y:
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
class UsdMeshShapeBuilder(UsdShapeBuilder):
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
    def _diffuse_texture_path(mesh_prim: Usd.Prim) -> Optional[str]:
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
    def _uv_coordinates(mesh_prim: Usd.Prim) -> Optional[NDArray[np.float64]]:
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


@dataclass
class USDParser(WorldModelParser):
    """
    Parses Universal Scene Description (USD) stages into worlds.

    A USD stage's arbitrary ``Xform`` nesting is scene organisation, not rigid-body
    structure: the rigid links of a physically articulated asset are the prims
    connected by its :class:`~pxr.UsdPhysics.Joint` prims (a joint's ``body0``/``body1``
    relationships), so those - not the raw prim hierarchy - are what is walked to build
    the kinematic tree. A stage with no physics joints at all is treated as one rigid
    body: every mesh and primitive shape under its root becomes a shape of a single
    root Body.

    .. note::
        Requires the ``usd-core`` package (``pxr``) to read the stage.

    .. note::
        Every rigid link is assumed to appear as the child (``body1``) of exactly one
        joint - a link that does not would be created (as another joint's parent) but
        never connected, and so would not appear in the parsed world.

    .. note::
        USD is right-handed. Unlike the axis convention, the up axis and the unit scale
        are not fixed: they are stage metadata (``UsdGeom.GetStageUpAxis``, typically Y
        or Z, and ``UsdGeom.GetStageMetersPerUnit``). This parser never assumes either -
        every transform it builds is relative to a prim's own parent, computed straight
        from the authored ``xformOpOrder``, so it comes out correct in whatever up axis
        and unit scale the stage itself declares.
    """

    stage: Usd.Stage
    """
    The USD stage to parse.

    Unlike URDF/SDF, a USD stage is not a flat block of source text one file happens to
    hold: composition (references, sublayers, ...) makes the opened stage itself, not
    any single file's text, the thing that fully describes what to parse. So this - not
    a file path - is the parser's payload field; :meth:`from_file` opens the file first.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    path_resolver: PathResolver = field(default_factory=CompositePathResolver)
    """
    The path resolver used for the asset references of this stage.
    """

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = Path(self.stage.GetRootLayer().identifier).stem

    # %% construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: Optional[str] = None,
        path_resolver: Optional[PathResolver] = None,
    ) -> USDParser:
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

    # %% entry point

    def parse(self) -> World:
        """
        Parses the stage into a world.

        :return: The parsed world.
        :raises UnsupportedUsdPhysicsJointTypeError: If the stage contains a physics
            joint of a type this parser does not build a Connection for.
        :raises UsdPhysicsJointMissingChildBodyError: If a physics joint's ``body1``
            relationship has no target.
        """
        joint_prims = [
            prim for prim in self.stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        ]
        if not joint_prims:
            return self._parse_jointless_stage()
        return self._parse_joint_graph(joint_prims)

    def _parse_joint_graph(self, joint_prims: List[Usd.Prim]) -> World:
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
        root_prim = self._root_prim()
        root_body_name = root_prim.GetName() if root_prim is not None else self.prefix
        world = World.create_with_root_body(root_body_name, self.prefix)
        root_body = world.root

        # Every joint is described (and so validated) before the world is touched
        # further: a World left partway through a failed modification is unusable, so
        # anything that can raise must run before entering another modify_world block.
        link_bodies: Dict[str, Body] = {}
        descriptions = [
            self._describe_joint(joint_prim, root_body, link_bodies)
            for joint_prim in joint_prims
        ]

        with world.modify_world():
            if root_prim is not None:
                self._attach_semantic_labels(world, root_prim, root_body)
            for path_string, link_body_instance in link_bodies.items():
                world.add_body(link_body_instance)
                self._attach_semantic_labels(
                    world, self.stage.GetPrimAtPath(path_string), link_body_instance
                )
            for description in descriptions:
                world.add_connection(self._create_connection(world, description))

        return world

    def _parse_jointless_stage(self) -> World:
        """
        Builds the world for a stage with no physics joints at all.

        Its root prim (see :meth:`_root_prim`) becomes a single-body World. With no
        identifiable root prim, there is nothing to unambiguously treat as "the" object
        either - so a synthetic root is created instead, and every top-level prim
        becomes its own body, attached to it with a
        :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`
        (freely posable, since the stage itself asserts no relationship between them).

        :return: The parsed world.
        """
        root_prim = self._root_prim()

        if root_prim is not None:
            world = World.create_with_root_body(root_prim.GetName(), self.prefix)
            root_body = world.root
            shapes = self._shapes_in_subtree(root_prim)
            shape_collection = ShapeCollection(shapes, reference_frame=root_body)
            shape_collection.transform_all_shapes_to_own_frame()
            root_body.visual = shape_collection
            root_body.collision = shape_collection
            with world.modify_world():
                self._attach_semantic_labels(world, root_prim, root_body)
            return world

        world = World.create_with_root_body(self.prefix, self.prefix)
        root_body = world.root
        with world.modify_world():
            for prim in self._top_level_prims():
                body = self._create_link_body(prim)
                world.add_body(body)
                self._attach_semantic_labels(world, prim, body)
                world.add_connection(
                    Connection6DoF.create_with_dofs(
                        world=world, parent=root_body, child=body
                    )
                )
        return world

    def _root_prim(self) -> Optional[Usd.Prim]:
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

    def _top_level_prims(self) -> List[Usd.Prim]:
        """
        :return: Every top-level prim of the stage (the pseudo-root's direct children).
        """
        return list(self.stage.GetPseudoRoot().GetChildren())

    # %% joints

    def _describe_joint(
        self,
        joint_prim: Usd.Prim,
        root_body: Body,
        link_bodies: Dict[str, Body],
    ) -> JointDescription:
        """
        Validates and describes the Connection one physics joint prim becomes, without
        touching a world - a World left partway through a failed modification is
        unusable, so every joint is described before any world modification begins.

        :param joint_prim: The USD physics joint prim (a Fixed/Revolute/PrismaticJoint).
        :param root_body: The world's root body, used as the connection's parent if the
            joint's ``body0`` relationship has no target.
        :param link_bodies: The link Bodies created so far, by stage path - extended in
            place as new links are resolved.
        :return: The description of the connection the joint becomes.
        :raises UnsupportedUsdPhysicsJointTypeError: If the joint's type has no
            Connection counterpart.
        :raises UsdPhysicsJointMissingChildBodyError: If the joint's ``body1``
            relationship has no target.
        """
        try:
            usd_joint_type = UsdPhysicsJointType(joint_prim.GetTypeName())
        except ValueError as error:
            raise UnsupportedUsdPhysicsJointTypeError(
                file_path=self.source_description,
                joint_path=str(joint_prim.GetPath()),
                joint_type=joint_prim.GetTypeName(),
                supported_types=list(UsdPhysicsJointType),
            ) from error
        connection_type = usd_joint_type.connection_type

        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body1_targets:
            raise UsdPhysicsJointMissingChildBodyError(
                file_path=self.source_description, joint_path=str(joint_prim.GetPath())
            )
        parent = (
            self._resolve_link_body(link_bodies, body0_targets[0])
            if body0_targets
            else root_body
        )
        child = self._resolve_link_body(link_bodies, body1_targets[0])

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

        if connection_type is FixedConnection:
            return JointDescription(
                connection_type=connection_type,
                parent=parent,
                child=child,
                parent_T_connection=parent_T_connection,
                connection_T_child=connection_T_child,
            )

        axis_joint = (
            UsdPhysics.RevoluteJoint(joint_prim)
            if connection_type is RevoluteConnection
            else UsdPhysics.PrismaticJoint(joint_prim)
        )
        axis = UsdAxis(axis_joint.GetAxisAttr().Get()).vector(reference_frame=parent)
        lower = axis_joint.GetLowerLimitAttr().Get()
        upper = axis_joint.GetUpperLimitAttr().Get()
        if connection_type is RevoluteConnection:
            lower, upper = math.radians(lower), math.radians(upper)
        if lower > upper:
            # Seen authored this way on a mirrored part (e.g. one blade of a pair of
            # scissors): the pair's joints share one axis convention, so the mirrored
            # joint's authored "lower"/"upper" swap relative to it even though both
            # describe the same-sized range of motion. The DOF's own lower/upper are
            # just its two extremes, so swapping the values (not negating them) keeps
            # the authored range of motion intact.
            lower, upper = upper, lower
        return JointDescription(
            connection_type=connection_type,
            parent=parent,
            child=child,
            parent_T_connection=parent_T_connection,
            connection_T_child=connection_T_child,
            axis=axis,
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=lower), upper=DerivativeMap(position=upper)
            ),
        )

    def _resolve_link_body(
        self, link_bodies: Dict[str, Body], prim_path: Sdf.Path
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
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.GetTypeName() == UsdGeomPrimType.MESH:
            # Most joints target the link's enclosing Xform, whose subtree holds its
            # shape(s); some instead target a link's mesh prim directly. Both resolve
            # to the same parent Xform, so a link is never split into two disconnected
            # bodies.
            prim = prim.GetParent()
        path_string = str(prim.GetPath())
        if path_string not in link_bodies:
            link_bodies[path_string] = self._create_link_body(prim)
        return link_bodies[path_string]

    @staticmethod
    def _create_connection(world: World, description: JointDescription) -> Connection:
        """
        Creates the Connection a joint description denotes, adding its degree of freedom
        to the world.

        :param world: The world the degree of freedom is added to.
        :param description: The description of the joint.
        :return: The connection describing the joint.
        """
        if description.connection_type is FixedConnection:
            return FixedConnection.create_with_dofs(
                world=world,
                name=description.name,
                parent=description.parent,
                child=description.child,
                parent_T_connection_expression=description.parent_T_connection,
                connection_T_child_expression=description.connection_T_child,
            )
        connection = description.connection_type.create_with_dofs(
            world=world,
            name=description.name,
            parent=description.parent,
            child=description.child,
            parent_T_connection_expression=description.parent_T_connection,
            connection_T_child_expression=description.connection_T_child,
            axis=description.axis,
            dof_limits=description.limits,
        )
        connection.dynamics = description.dynamics
        return connection

    # %% links and shapes

    def _create_link_body(self, link_prim: Usd.Prim) -> Body:
        """
        Creates the Body for one rigid link, with a Shape for every mesh/primitive in
        its USD subtree and its :class:`~pxr.UsdPhysics.MassAPI` inertial properties, if
        applied.

        :param link_prim: The link's root USD prim.
        :return: The created body, not yet added to a world.
        """
        shapes = self._shapes_in_subtree(link_prim)
        shape_collection = ShapeCollection(shapes)
        body = Body(
            name=PrefixedName(link_prim.GetName(), self.prefix),
            visual=shape_collection,
            collision=shape_collection,
        )
        inertial = self._parse_inertial(link_prim, body)
        if inertial is not None:
            body.inertial = inertial
        return body

    def _shapes_in_subtree(self, link_prim: Usd.Prim) -> List[Shape]:
        """
        Creates the Shape for every mesh/primitive prim in a link's subtree.

        :param link_prim: The link the shapes are positioned relative to, and the root
            of the subtree to search for them.
        :return: The created shapes.
        """
        link_to_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        shapes = []
        for prim in Usd.PrimRange(link_prim):
            shape = self._create_shape(prim, link_to_world)
            if shape is not None:
                shapes.append(shape)
        return shapes

    def _create_shape(
        self, prim: Usd.Prim, link_to_world: Gf.Matrix4d
    ) -> Optional[Shape]:
        """
        Creates the Shape a mesh/primitive prim describes.

        A prim that is not a renderable geometric primitive at all (a plain ``Xform``
        grouping node, a camera, a shader, ...) is not shape geometry and is silently
        skipped; one that is (:class:`~pxr.UsdGeom.Gprim`) but of a type this parser
        does not build a Shape for (e.g. ``Cone``, ``Capsule``) raises instead, the same
        way an unrecognised joint type does, rather than silently vanishing from the
        built world.

        :param prim: The prim to create a shape for.
        :param link_to_world: The enclosing link's local-to-world transform.
        :return: The created shape, or ``None`` if ``prim`` is not shape geometry.
        :raises UnsupportedUsdGeometryTypeError: If ``prim`` is a renderable geometric
            primitive of a type this parser does not build a Shape for.
        """
        type_name = prim.GetTypeName()
        try:
            geom_type = UsdGeomPrimType(type_name)
        except ValueError as error:
            if prim.IsA(UsdGeom.Gprim):
                raise UnsupportedUsdGeometryTypeError(
                    file_path=self.source_description,
                    prim_path=str(prim.GetPath()),
                    geometry_type=type_name,
                    supported_types=list(UsdGeomPrimType),
                ) from error
            return None
        return geom_type.create_shape(prim, link_to_world)

    # %% inertials

    @staticmethod
    def _parse_inertial(link_prim: Usd.Prim, body: Body) -> Optional[Inertial]:
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

    @staticmethod
    def _attach_semantic_labels(world: World, prim: Usd.Prim, body: Body) -> None:
        """
        Attaches a :class:`UsdSemanticLabels` annotation to ``body`` for every
        ``UsdSemantics.LabelsAPI`` taxonomy directly authored on ``prim``, if any.

        Must be called inside ``world``'s modification context, alongside adding
        ``body`` itself.

        :param world: The world to add the annotation to.
        :param prim: The USD prim ``body`` was built from.
        :param body: The already-added body the annotation's root is.
        """
        for taxonomy, labels in USDParser._read_semantic_labels(prim).items():
            world.add_semantic_annotation(
                UsdSemanticLabels(root=body, taxonomy=taxonomy, labels=labels)
            )

    @staticmethod
    def _read_semantic_labels(prim: Usd.Prim) -> Dict[str, List[str]]:
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
