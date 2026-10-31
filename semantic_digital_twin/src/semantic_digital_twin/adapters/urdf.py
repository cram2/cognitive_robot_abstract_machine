from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from urdf_parser_py import urdf as urdfpy
from xacro import process_file

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.world_model_parser import WorldModelParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    NegativeConnectionVelocity,
    MissingURDFJointAxisError,
    ZeroLengthURDFJointAxisError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.utils import (
    suppress_stdout_stderr,
    hacky_urdf_parser_fix,
    robot_name_from_urdf_string,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Sphere,
    Cylinder,
    Mesh,
    Scale,
    Color,
    Shape,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Connection

connection_type_map = {  # 'unknown': JointType.UNKNOWN,
    "revolute": RevoluteConnection,
    "continuous": RevoluteConnection,
    "prismatic": PrismaticConnection,
    "floating": Connection6DoF,
    # 'planar': JointType.PLANAR,
    "fixed": FixedConnection,
}


def urdf_joint_axis(urdf_joint: urdfpy.Joint) -> list[float]:
    """
    The axis a URDF joint moves along, as the unit vector the connections expect.

    URDF normalizes the axis it is given. A joint that declares no axis is rejected
    rather than given a default, so a description is read as written.

    :param urdf_joint: The URDF joint to read the axis of.
    :return: The joint's axis as a unit vector.
    """
    if urdf_joint.axis is None:
        raise MissingURDFJointAxisError(
            joint_name=urdf_joint.name, joint_type=urdf_joint.type
        )
    axis = np.asarray(urdf_joint.axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ZeroLengthURDFJointAxisError(joint_name=urdf_joint.name)
    return (axis / norm).tolist()


def urdf_joint_to_limits(
    urdf_joint: urdfpy.Joint,
) -> DegreeOfFreedomLimits[float]:
    """
    Maps the URDF joint specifications to lower and upper joint limits, including
    position and velocity constraints. Mimics and safety controller parameters are also
    considered when determining the limits.

    :param urdf_joint: A URDF (Unified Robot Description Format) joint object which
        contains the joint's type, limits, safety controller, and mimic information.
    :return: The limits of the joint's position and velocity.
    """
    lower = None
    upper = None
    limit = getattr(urdf_joint, "limit", None)
    if not urdf_joint.type == "continuous":
        lower = limit.lower if limit is not None else None
        upper = limit.upper if limit is not None else None

        safety_controller = getattr(urdf_joint, "safety_controller", None)
        lower = (
            max(safety_controller.soft_lower_limit, limit.lower)
            if safety_controller is not None and limit is not None
            else lower
        )
        upper = (
            min(safety_controller.soft_upper_limit, limit.upper)
            if safety_controller is not None and limit is not None
            else upper
        )

    velocity = getattr(limit, "velocity", None) if limit is not None else None

    if velocity is not None and velocity < 0:
        raise NegativeConnectionVelocity(
            connection_name=urdf_joint.name, velocity=velocity
        )
    return DegreeOfFreedomLimits.from_position_range_and_speed(
        lower_position=lower, upper_position=upper, maximum_speed=velocity
    )


@dataclass
class URDFParser(WorldModelParser):
    """
    Class to parse URDF files to worlds.

    Must set either urdf or file_path.
    """

    urdf: str
    """
    The URDF string.
    """

    prefix: str | None = None
    """
    The prefix for every name used in this world.
    """

    path_resolver: PathResolver = field(default_factory=CompositePathResolver)
    """
    The path resolver to use for resolving URIs in the URDF file.
    """

    def __post_init__(self):
        self.urdf = hacky_urdf_parser_fix(self.urdf)
        self.parsed = urdfpy.URDF.from_xml_string(self.urdf)
        if self.prefix is None:
            self.prefix = robot_name_from_urdf_string(self.urdf)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        use_visual_as_collision_backup: bool = False,
    ) -> URDFParser:
        if file_path.endswith(".xacro"):
            return cls.from_xacro(
                file_path,
                prefix,
                use_visual_as_collision_backup=use_visual_as_collision_backup,
            )

        path_resolver = path_resolver or CompositePathResolver()

        file_path = path_resolver.resolve(file_path)
        if file_path is not None:
            with open(file_path, "r") as file:
                # Since parsing URDF causes a lot of warning messages which can't be deactivated, we suppress them
                with suppress_stdout_stderr():
                    urdf = file.read()
        urdf_parser = cls(
            urdf=urdf,
            prefix=prefix,
            use_visual_as_collision_backup=use_visual_as_collision_backup,
        )
        urdf_parser.path_resolver = path_resolver
        return urdf_parser

    @classmethod
    def from_xacro(
        cls,
        xacro_path: str,
        prefix: str | None = None,
        mappings: dict[str, str] | None = None,
        use_visual_as_collision_backup: bool = False,
    ) -> URDFParser:
        """
        Creates a parser from a xacro file by expanding it to URDF.

        The xacro file is resolved and processed into a URDF string, applying the given
        substitution arguments, before constructing the parser.

        :param xacro_path: The path to the xacro file to expand.
        :param prefix: The prefix for every name used in this world.
        :param mappings: The xacro substitution arguments to apply during expansion (the
            ``arg`` values, e.g. ``{"ur_type": "ur5"}``).
        :param use_visual_as_collision_backup: Whether a link that describes no
            collision geometry collides with its visual geometry instead.
        :return: A parser for the world described by the expanded xacro file.
        """
        xacro_path = CompositePathResolver().resolve(xacro_path)
        urdf = process_file(xacro_path, mappings=mappings).toxml()
        return URDFParser(
            urdf=urdf,
            prefix=prefix,
            use_visual_as_collision_backup=use_visual_as_collision_backup,
        )

    def parse(self) -> World:
        for joint in self.parsed.joints:
            if connection_type_map.get(joint.type) in (
                RevoluteConnection,
                PrismaticConnection,
            ):
                urdf_joint_axis(joint)

        prefix = self.parsed.name
        links = [
            self.parse_link(link, PrefixedName(link.name, prefix))
            for link in self.parsed.links
        ]
        root = [link for link in links if link.name.name == self.parsed.get_root()][0]
        world = World()
        world.name = self.prefix
        with world.modify_world():
            world.add_kinematic_structure_entity(root)
            main_joints = []
            mimic_joints = []

            for joint in self.parsed.joints:
                if joint.mimic is not None:
                    mimic_joints.append(joint)
                else:
                    main_joints.append(joint)

            parsed_joints = []
            for joint in main_joints + mimic_joints:
                parent = [link for link in links if link.name.name == joint.parent][0]
                child = [link for link in links if link.name.name == joint.child][0]
                parsed_joint = self.parse_joint(joint, parent, child, world, prefix)
                parsed_joints.append(parsed_joint)

            [world.add_connection(joint) for joint in parsed_joints]

        return world

    def parse_joint(
        self, joint: urdfpy.Joint, parent: Body, child: Body, world: World, prefix: str
    ) -> Connection:
        """
        Parses a given URDF joint and creates a corresponding connection object.

        The function processes the provided joint data, extracting necessary information
        including translation offsets, rotation offsets, connection type, and relevant
        joint limits. It maps URDF joint types to predefined connection types and either
        retrieves or creates a degree of freedom (DOF) in the world context. It
        generates and returns a connection object representing the relationship between
        a parent and a child body.

        :param joint: The URDF joint to be parsed.
        :param parent: The parent body to be connected by the joint.
        :param child: The child body to be connected by the joint.
        :param world: The world instance containing degrees of freedom.
        :param prefix: The prefix for naming connections and DOFs.
        :return: A connection object representing the parsed joint.
        """
        connection_name = PrefixedName(joint.name, prefix)
        connection_type = connection_type_map.get(joint.type, Connection)
        translation_offset = getattr(joint.origin, "xyz", [0, 0, 0])
        rotation_offset = getattr(joint.origin, "rpy", [0, 0, 0])

        parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_offset[0],
            y=translation_offset[1],
            z=translation_offset[2],
            roll=rotation_offset[0],
            pitch=rotation_offset[1],
            yaw=rotation_offset[2],
        )
        if connection_type is FixedConnection:
            return FixedConnection(
                name=connection_name,
                parent=parent,
                child=child,
                parent_T_connection_expression=parent_T_connection,
            )
        if connection_type is Connection6DoF:
            return Connection6DoF.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                name=connection_name,
                parent_T_connection_expression=parent_T_connection,
            )

        dof_name = connection_name
        multiplier, offset = 1.0, 0.0
        if joint.mimic:
            multiplier = (
                joint.mimic.multiplier if joint.mimic.multiplier is not None else 1
            )
            offset = joint.mimic.offset if joint.mimic.offset is not None else 0
            dof_name = PrefixedName(joint.mimic.joint, prefix)

        if dof_name not in [d.name for d in world.degrees_of_freedom]:
            dof = DegreeOfFreedom(name=dof_name, limits=urdf_joint_to_limits(joint))
            world.add_degree_of_freedom(dof)
        else:
            dof = world.get_degree_of_freedom_by_name(dof_name)

        result = connection_type(
            name=connection_name,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection,
            multiplier=multiplier,
            offset=offset,
            axis=Vector3(*urdf_joint_axis(joint), reference_frame=parent),
            raw_dof=dof,
            dynamics=self.parse_dynamics(joint),
        )
        return result

    def parse_dynamics(self, joint: urdfpy.Joint) -> JointDynamics:
        """
        Parses the dynamic properties of a URDF joint.

        Properties the joint leaves undeclared keep their default. URDF has no notion of
        an armature, which models the rotor inertia of a transmission, so that one is
        always left to the simulator.

        :param joint: The URDF joint whose ``dynamics`` element is parsed.
        :return: The dynamic properties of the joint.
        """
        if joint.dynamics is None:
            return JointDynamics()

        damping = joint.dynamics.damping
        dry_friction = joint.dynamics.friction
        return JointDynamics(
            damping=damping if damping is not None else 0.0,
            dry_friction=dry_friction if dry_friction is not None else 0.0,
        )

    def parse_link(self, link: urdfpy.Link, parent_frame: PrefixedName) -> Body:
        """
        Parses a URDF link to a link object.

        :param link: The URDF link to parse.
        :param parent_frame: The parent frame of the link, used for transformations of
            collisions and visuals.
        :return: The parsed link object.
        """
        name = PrefixedName(prefix=self.prefix, name=link.name)
        body = Body(name=name)
        visuals = self.parse_geometry(link.visuals, body)
        collisions = self.parse_geometry(link.collisions, body)
        if not collisions.shapes and self.use_visual_as_collision_backup:
            collisions = self.parse_geometry(link.visuals, body)
        body.visual = visuals
        body.collision = collisions
        inertial = self.parse_inertial(link, body)
        if inertial is not None:
            body.inertial = inertial
        return body

    def parse_inertial(self, link: urdfpy.Link, body: Body) -> Inertial | None:
        """
        Parses the inertial properties of a URDF link.

        URDF expresses the inertia tensor in the inertial frame of the link, so it is
        rotated into the link frame, which is the frame :class:`Inertial` expects.

        :param link: The URDF link whose ``inertial`` element is parsed.
        :param body: The body the properties belong to, used as their reference frame.
        :return: The inertial properties, or ``None`` if the link declares none.
        """
        if link.inertial is None:
            return None

        origin = link.inertial.origin
        center_of_mass = origin.xyz if origin is not None else [0.0, 0.0, 0.0]
        roll_pitch_yaw = origin.rpy if origin is not None else [0.0, 0.0, 0.0]

        urdf_inertia = link.inertial.inertia
        inertia_in_inertial_frame = InertiaTensor.from_values(
            ixx=urdf_inertia.ixx,
            iyy=urdf_inertia.iyy,
            izz=urdf_inertia.izz,
            ixy=urdf_inertia.ixy,
            ixz=urdf_inertia.ixz,
            iyz=urdf_inertia.iyz,
        )
        link_R_inertial = RotationMatrix.from_rpy(*roll_pitch_yaw).to_np()[:3, :3]
        inertia = link_R_inertial @ inertia_in_inertial_frame.data @ link_R_inertial.T
        inertia_in_link_frame = InertiaTensor(data=inertia)

        return Inertial(
            mass=link.inertial.mass,
            center_of_mass=Point3(*center_of_mass, reference_frame=body),
            inertia=inertia_in_link_frame,
        )

    def parse_geometry(
        self,
        geometry: list[urdfpy.Collision] | list[urdfpy.Visual],
        body: Body,
    ) -> ShapeCollection:
        """
        Parses a URDF geometry to the corresponding shapes.

        :param geometry: The URDF geometry to parse either the collisions of visuals.'
        :param body: The body of the geometry, used for back referencing.
        :return: A List of shapes corresponding to the URDF geometry.
        """
        res = []
        for i, geom in enumerate(geometry):
            params = (*(geom.origin.xyz + geom.origin.rpy),) if geom.origin else ()
            origin_transform = HomogeneousTransformationMatrix.from_xyz_rpy(
                *params, reference_frame=body
            )
            if isinstance(geom.geometry, urdfpy.Box):
                shape = Box(origin=origin_transform, scale=Scale(*geom.geometry.size))
            elif isinstance(geom.geometry, urdfpy.Sphere):
                shape = Sphere(origin=origin_transform, radius=geom.geometry.radius)
            elif isinstance(geom.geometry, urdfpy.Cylinder):
                shape = Cylinder(
                    origin=origin_transform,
                    width=geom.geometry.radius * 2,
                    height=geom.geometry.length,
                )
            elif isinstance(geom.geometry, urdfpy.Mesh):
                if geom.geometry.filename is None:
                    raise ValueError("Mesh geometry must have a filename.")
                shape = Mesh(
                    origin=origin_transform,
                    filename=self.path_resolver.resolve(geom.geometry.filename),
                    scale=Scale(*(geom.geometry.scale or (1, 1, 1))),
                )
            else:
                continue
            color = self.color_of(geom, shape)
            if color is not None:
                shape.color = color
            res.append(shape)
        return ShapeCollection(res, reference_frame=body)

    def color_of(
        self, geometry_element: urdfpy.Visual | urdfpy.Collision, shape: Shape
    ) -> Color | None:
        """
        :param geometry_element: A visual or collision element of the description.
        :param shape: The shape parsed from the element.
        :return: The color the element's material gives the shape: the material's own
            color, else the color of the description's material of that name. ``None``
            if the element gives it no color, as a collision element and an element
            without a colored material do, and for a mesh whose file paints itself,
            which keeps its own look as a viewer of the description shows it.
        """
        if not isinstance(geometry_element, urdfpy.Visual):
            return None
        material = geometry_element.material
        if material is None:
            return None
        if material.color is not None:
            color = Color.from_rgba(*material.color.rgba)
        else:
            named_colors = {
                named.name: named.color
                for named in self.parsed.materials
                if named.color is not None
            }
            if material.name not in named_colors:
                return None
            color = Color.from_rgba(*named_colors[material.name].rgba)
        # The mesh's file is read only when there is a color to give.
        if isinstance(shape, Mesh) and shape.is_textured:
            return None
        return color
