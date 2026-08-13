"""
Serializing a :class:`~semantic_digital_twin.world.World` as a self-contained URDF.

Any adapter that resolves a robot description into a :class:`World` (Gazebo/SDF, MJCF,
...) can bundle it by parsing it and handing the result to
:meth:`UrdfDocument.of_world`; this module walks the kinematic tree and serializes it,
it has no notion of the source format.
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field

from coraplex.datastructures.enums import JointType
from scipy.spatial.transform import Rotation
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection,
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import ClassVar, Dict, Iterable, List, Optional, Type

from cramera.onboard.bundle_urdf import BundledAssets, BundleReport


@dataclass
class UrdfDocument:
    """
    A URDF document being assembled from a world's kinematic tree.
    """

    CONNECTION_JOINT_TYPES: ClassVar[Dict[Type[Connection], JointType]] = {
        FixedConnection: JointType.FIXED,
        RevoluteConnection: JointType.REVOLUTE,
        PrismaticConnection: JointType.PRISMATIC,
        Connection6DoF: JointType.FLOATING,
        OmniDrive: JointType.FLOATING,
    }
    """
    The joint type a connection class becomes.

    A :class:`RevoluteConnection` additionally becomes :attr:`JointType.CONTINUOUS` when its degree of freedom has no position
    limits.
    """

    SYNTHESIZED_ROOT_LINK: ClassVar[str] = "world_root"
    """
    Name of the link :meth:`of_bodies` roots its document in, which belongs to no body
    of the world it serializes part of.
    """

    COORDINATE_PRECISION: ClassVar[int] = 6
    """
    Decimal places a bundled numeric attribute keeps.
    """

    AXIS_JOINT_TYPES: ClassVar[frozenset] = frozenset(
        {JointType.REVOLUTE, JointType.CONTINUOUS, JointType.PRISMATIC}
    )
    """
    Joint types that carry an ``axis`` element.
    """

    LIMITED_JOINT_TYPES: ClassVar[frozenset] = frozenset(
        {JointType.REVOLUTE, JointType.PRISMATIC}
    )
    """
    Joint types that carry a ``limit`` element when their degree of freedom is limited.
    """

    output_directory: str
    """
    Directory the URDF and its ``meshes/`` tree are written to.
    """

    mesh_subdirectory: str
    """
    Directory bundled meshes nest under, so meshes from different source formats or
    models cannot collide.
    """

    root_element: ElementTree.Element
    """
    The document's ``robot`` element, which every link and joint is added to.
    """

    assets: BundledAssets
    """
    Collects the files copied into the bundle.
    """

    joint_names: List[str] = field(default_factory=list)
    """
    Names of the joints added so far, in document order.
    """

    movable_joint_names: List[str] = field(default_factory=list)
    """
    Names of the added joints that are not fixed.
    """

    @classmethod
    def of_world(
        cls, world: World, name: str, output_directory: str, mesh_subdirectory: str
    ) -> BundleReport:
        """
        Serialize a parsed world, with every mesh it references, as a URDF.

        :param world: The world to serialize, already resolved to concrete shapes and
            poses by whichever adapter parsed it.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree go into.
        :param mesh_subdirectory: Directory bundled meshes nest under.
        """
        os.makedirs(output_directory, exist_ok=True)
        document = cls(
            output_directory=output_directory,
            mesh_subdirectory=mesh_subdirectory,
            root_element=ElementTree.Element("robot", {"name": name}),
            assets=BundledAssets(bundle_root=output_directory),
        )
        bodies = world.bodies_topologically_sorted
        for body in bodies:
            document.add_link(body)
            if body.parent_connection is not None:
                document.add_joint(body.parent_connection)
        return document.write(name, bodies)

    @classmethod
    def of_bodies(
        cls,
        bodies: List[Body],
        name: str,
        output_directory: str,
        mesh_subdirectory: str,
        identity_root: Optional[Body] = None,
    ) -> BundleReport:
        """
        Serialize part of a world -- the bodies no parsed source describes -- as a URDF.

        Connections *within* the subset are kept, so a drawer stays prismatic and keeps
        following its recorded positions. A body whose parent lies outside the subset
        has no joint to inherit and is fixed to :attr:`SYNTHESIZED_ROOT_LINK` at the
        pose it holds in the world, which is also what leaves the document with a single
        root.

        :param bodies: The bodies to serialize, in the order they should appear.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree go into.
        :param mesh_subdirectory: Directory bundled meshes nest under.
        :param identity_root: Body grafted at the origin instead of at its world pose --
            a robot subtree's base, whose live pose the viewer applies on top of the
            model.
        """
        os.makedirs(output_directory, exist_ok=True)
        document = cls(
            output_directory=output_directory,
            mesh_subdirectory=mesh_subdirectory,
            root_element=ElementTree.Element("robot", {"name": name}),
            assets=BundledAssets(bundle_root=output_directory),
        )
        ElementTree.SubElement(
            document.root_element, "link", {"name": cls.SYNTHESIZED_ROOT_LINK}
        )
        serialized = {str(body.name) for body in bodies}
        for body in bodies:
            document.add_link(body)
            connection = body.parent_connection
            if body is identity_root:
                document.graft_onto_root(body, pose=HomogeneousTransformationMatrix())
            elif (
                connection is not None
                and str(connection.parent.name) in serialized
                and cls.supports(connection)
            ):
                document.add_joint(connection)
            else:
                document.graft_onto_root(body)
        return document.write(name, bodies)

    def graft_onto_root(
        self, body: Body, pose: Optional[HomogeneousTransformationMatrix] = None
    ) -> None:
        """
        Fix a body to :attr:`SYNTHESIZED_ROOT_LINK`.

        :param body: The body to attach, whose own parent this document does not
            contain.
        :param pose: The pose the body is fixed at; the pose it holds in its world when
            not given.
        """
        joint_element = ElementTree.SubElement(
            self.root_element,
            "joint",
            {
                "name": "%s_to_%s" % (self.SYNTHESIZED_ROOT_LINK, str(body.name)),
                "type": JointType.FIXED.name.lower(),
            },
        )
        ElementTree.SubElement(
            joint_element, "parent", {"link": self.SYNTHESIZED_ROOT_LINK}
        )
        ElementTree.SubElement(joint_element, "child", {"link": str(body.name)})
        self._set_origin(joint_element, pose if pose is not None else body.global_pose)
        self.joint_names.append(joint_element.attrib["name"])

    def write(self, name: str, bodies: Iterable[Body]) -> BundleReport:
        """
        Write the assembled document to disk and report what it contains.

        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param bodies: The bodies serialized into the document.
        """
        urdf_out = os.path.join(self.output_directory, "%s.urdf" % name)
        ElementTree.indent(self.root_element)
        ElementTree.ElementTree(self.root_element).write(
            urdf_out, encoding="utf-8", xml_declaration=True
        )
        return BundleReport(
            name=name,
            urdf=urdf_out,
            source=urdf_out,
            links=[str(body.name) for body in bodies],
            joints=self.joint_names,
            movable_joints=self.movable_joint_names,
            meshes_copied=len(self.assets.copied),
            mesh_suffixes=self.assets.mesh_suffixes,
            references_rewritten=len(self.assets.copied),
            missing=self.assets.missing,
        )

    # %% links
    def add_link(self, body: Body) -> None:
        """
        Add a ``link`` element for a body, with one ``visual`` per shape it carries.

        :param body: The body the link describes.
        """
        link_element = ElementTree.SubElement(
            self.root_element, "link", {"name": str(body.name)}
        )
        for shape in body.visual.shapes:
            visual_element = ElementTree.SubElement(link_element, "visual")
            self._set_origin(visual_element, shape.origin)
            self._add_geometry(visual_element, shape)
            self._add_material(visual_element, shape)

    def _add_geometry(self, visual_element: ElementTree.Element, shape: Shape) -> None:
        """
        Add the ``geometry`` a shape describes, copying a mesh's file into the bundle
        first if the shape is one.

        :param visual_element: The ``visual`` element the geometry belongs to.
        :param shape: The shape to describe.
        :raises TypeError: If the shape is of a type this bundler does not support.
        """
        geometry_element = ElementTree.SubElement(visual_element, "geometry")
        if isinstance(shape, Box):
            ElementTree.SubElement(
                geometry_element,
                "box",
                {"size": self._format_numbers(shape.scale.to_np())},
            )
            return
        if isinstance(shape, Sphere):
            ElementTree.SubElement(
                geometry_element, "sphere", {"radius": str(shape.radius)}
            )
            return
        if isinstance(shape, Cylinder):
            ElementTree.SubElement(
                geometry_element,
                "cylinder",
                {"radius": str(shape.radius), "length": str(shape.height)},
            )
            return
        if not isinstance(shape, Mesh):
            raise TypeError("Unsupported shape type for bundling: %s" % type(shape))

        relative_path = os.path.join(
            self.mesh_subdirectory,
            os.path.basename(os.path.dirname(shape.filename)),
            os.path.basename(shape.filename),
        )
        bundled = os.path.join(self.output_directory, "meshes", relative_path)
        if self.assets.copy(shape.filename, bundled):
            self.assets.copy_side_assets(shape.filename, bundled)
        ElementTree.SubElement(
            geometry_element,
            "mesh",
            {
                "filename": "meshes/" + relative_path.replace(os.sep, "/"),
                "scale": self._format_numbers(shape.scale.to_np()),
            },
        )

    def _add_material(self, visual_element: ElementTree.Element, shape: Shape) -> None:
        """
        Add the ``material`` a shape's colour describes.

        :param visual_element: The ``visual`` element the material belongs to.
        :param shape: The shape whose colour is described.
        """
        material_element = ElementTree.SubElement(
            visual_element, "material", {"name": ""}
        )
        color = shape.color
        ElementTree.SubElement(
            material_element,
            "color",
            {"rgba": self._format_numbers([color.R, color.G, color.B, color.A])},
        )

    # %% joints
    def add_joint(self, connection: Connection) -> None:
        """
        Add a ``joint`` element for a connection.

        :param connection: The connection the joint describes.
        """
        joint_type = self._joint_type(connection)
        joint_element = ElementTree.SubElement(
            self.root_element,
            "joint",
            {"name": str(connection.name), "type": joint_type.name.lower()},
        )
        ElementTree.SubElement(
            joint_element, "parent", {"link": str(connection.parent.name)}
        )
        ElementTree.SubElement(
            joint_element, "child", {"link": str(connection.child.name)}
        )
        self._set_origin(joint_element, self._joint_origin(connection, joint_type))

        if joint_type in self.AXIS_JOINT_TYPES:
            ElementTree.SubElement(
                joint_element,
                "axis",
                {"xyz": self._format_numbers(connection.axis.to_np()[:3])},
            )
        if (
            joint_type in self.LIMITED_JOINT_TYPES
            and connection.dof.has_position_limits()
        ):
            limits = connection.dof.limits
            ElementTree.SubElement(
                joint_element,
                "limit",
                {
                    "lower": str(limits.lower.position),
                    "upper": str(limits.upper.position),
                    "velocity": str(limits.upper.velocity or 0.0),
                    "effort": "0.0",
                },
            )
        self.joint_names.append(str(connection.name))
        if joint_type is not JointType.FIXED:
            self.movable_joint_names.append(str(connection.name))

    @classmethod
    def _joint_origin(
        cls, connection: Connection, joint_type: JointType
    ) -> HomogeneousTransformationMatrix:
        """
        The parent-to-child pose a joint's ``origin`` states.

        URDF reads a joint as its origin followed by the joint's own displacement, so a
        joint whose displacement is supplied from outside -- the axis-driven types,
        which a recording or a live bridge drives -- must be written at its zero. Its
        :attr:`Connection.origin` is the pose at the *current* value, which would bake
        that value in and have the supplied one applied on top of it. Every other joint
        type carries no value of its own, so its full origin is the only thing placing
        its child.

        :param connection: The connection the joint describes.
        :param joint_type: The joint type the connection becomes.
        """
        if joint_type in cls.AXIS_JOINT_TYPES:
            return connection.parent_T_connection_expression
        return connection.origin

    @classmethod
    def supports(cls, connection: Connection) -> bool:
        """
        Whether a connection maps onto a URDF joint type.

        A body behind an unsupported connection is grafted onto the document root at its
        world pose instead.

        :param connection: The connection to check.
        """
        return isinstance(connection, tuple(cls.CONNECTION_JOINT_TYPES))

    @classmethod
    def _joint_type(cls, connection: Connection) -> JointType:
        """
        The joint type a connection becomes.

        :param connection: The connection to classify.
        :raises TypeError: If the connection is of a type this bundler does not support.
        """
        if (
            isinstance(connection, RevoluteConnection)
            and not connection.dof.has_position_limits()
        ):
            return JointType.CONTINUOUS
        for connection_type, joint_type in cls.CONNECTION_JOINT_TYPES.items():
            if isinstance(connection, connection_type):
                return joint_type
        raise TypeError(
            "Unsupported connection type for bundling: %s" % type(connection)
        )

    # %% numeric formatting
    @classmethod
    def _set_origin(
        cls, element: ElementTree.Element, pose: HomogeneousTransformationMatrix
    ) -> None:
        """
        Add an ``origin`` child expressing a pose as URDF does: a translation plus a
        fixed-axis (extrinsic) roll-pitch-yaw rotation.

        :param element: The element the origin belongs to.
        :param pose: The pose to express, relative to the frame the element implies.
        """
        matrix = pose.to_np()
        with warnings.catch_warnings():
            # at ±90° pitch the Euler decomposition is not unique, but scipy's choice
            # still reproduces the rotation exactly — the warning is pure noise here
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            roll, pitch, yaw = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
        ElementTree.SubElement(
            element,
            "origin",
            {
                "xyz": cls._format_numbers(matrix[:3, 3]),
                "rpy": cls._format_numbers([roll, pitch, yaw]),
            },
        )

    @classmethod
    def _format_numbers(cls, values: Iterable[float]) -> str:
        """
        Numbers as the space-separated attribute value a URDF carries.

        :param values: The numbers to format.
        """
        return " ".join(
            str(round(float(value), cls.COORDINATE_PRECISION)) for value in values
        )
