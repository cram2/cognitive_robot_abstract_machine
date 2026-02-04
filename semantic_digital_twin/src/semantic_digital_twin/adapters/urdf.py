from __future__ import annotations

import os
from dataclasses import dataclass

from typing_extensions import Optional, Tuple, Union, List, Dict
from urdf_parser_py import urdf as urdfpy

from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import ParsingError
from ..spatial_types.derivatives import Derivatives, DerivativeMap
from ..spatial_types.spatial_types import HomogeneousTransformationMatrix, Vector3
from ..utils import (
    suppress_stdout_stderr,
    hacky_urdf_parser_fix,
    robot_name_from_urdf_string,
)
from ..world import World
from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom, DegreeOfFreedomLimits
from ..world_description.geometry import (
    Box,
    Sphere,
    Cylinder,
    FileMesh,
    Scale,
    Color,
)
from ..world_description.shape_collection import ShapeCollection
from ..world_description.world_entity import Body, Connection

connection_type_map = {  # 'unknown': JointType.UNKNOWN,
    "revolute": RevoluteConnection,
    "continuous": RevoluteConnection,
    "prismatic": PrismaticConnection,
    # 'floating': JointType.FLOATING,
    # 'planar': JointType.PLANAR,
    "fixed": FixedConnection,
}


def urdf_joint_to_limits(
    urdf_joint: urdfpy.Joint,
) -> Tuple[DerivativeMap[float], DerivativeMap[float]]:
    """
    Maps the URDF joint specifications to lower and upper joint limits, including
    position and velocity constraints. Mimics and safety controller parameters are
    also considered when determining the limits.

    :param urdf_joint: A URDF (Unified Robot Description Format) joint object
                       which contains the joint's type, limits, safety controller,
                       and mimic information.
    :return: A tuple containing two DerivativeMap objects, representing the lower
             and upper limits of the joint in terms of position and velocity.
    """
    lower_limits = DerivativeMap()
    upper_limits = DerivativeMap()
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

        lower_limits.position = lower
        upper_limits.position = upper

    velocity = getattr(limit, "velocity", None) if limit is not None else None
    lower_limits.velocity = -velocity if velocity is not None else None
    upper_limits.velocity = velocity if velocity is not None else None

    if urdf_joint.mimic is not None:
        multiplier = (
            urdf_joint.mimic.multiplier
            if urdf_joint.mimic.multiplier is not None
            else 1
        )
        offset = urdf_joint.mimic.offset if urdf_joint.mimic.offset is not None else 0

        for d2 in Derivatives.range(Derivatives.position, Derivatives.velocity):
            lower_limits.data[d2] -= offset
            upper_limits.data[d2] -= offset
            if multiplier < 0:
                upper_limits.data[d2], lower_limits.data[d2] = (
                    lower_limits.data[d2],
                    upper_limits.data[d2],
                )
            upper_limits.data[d2] /= multiplier
            lower_limits.data[d2] /= multiplier
    return lower_limits, upper_limits


@dataclass
class URDFParser:
    """
    Class to parse URDF files to worlds.
    Must set either urdf or file_path.
    """

    urdf: str
    """
    The URDF string.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    package_resolver: Optional[Dict[str, str]] = None
    """
    The package resolver to use for resolving package paths in the URDF file. If ROS is installed, ROS will be used
     to resolve the paths, otherwise the package_resolver must be provided if the URDF file contains package paths.
     The key is the package name and the value is the path to the package. You can also set the environment variable
     `ROS_PACKAGE_PATH` to a colon-separated list of package paths, which will be used as the package resolver.
    """

    def __post_init__(self):
        self.urdf = hacky_urdf_parser_fix(self.urdf)
        self.parsed = urdfpy.URDF.from_xml_string(self.urdf)
        if self.prefix is None:
            self.prefix = robot_name_from_urdf_string(self.urdf)
        if self.package_resolver is None:
            package_paths = os.environ.get("ROS_PACKAGE_PATH", "").split(":")
            self.package_resolver = {
                os.path.basename(path): path
                for path in package_paths
                if os.path.exists(path)
            }

    @classmethod
    def from_file(cls, file_path: str, prefix: Optional[str] = None) -> URDFParser:
        if file_path is not None:
            with open(file_path, "r") as file:
                # Since parsing URDF causes a lot of warning messages which can't be deactivated, we suppress them
                with suppress_stdout_stderr():
                    urdf = file.read()
        return URDFParser(urdf=urdf, prefix=prefix)

    def parse(self) -> World:
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
            joints = []
            for joint in self.parsed.joints:
                parent = [link for link in links if link.name.name == joint.parent][0]
                child = [link for link in links if link.name.name == joint.child][0]
                parsed_joint = self.parse_joint(joint, parent, child, world, prefix)
                joints.append(parsed_joint)

            [world.add_connection(joint) for joint in joints]

        return world

    def parse_joint(
        self, joint: urdfpy.Joint, parent: Body, child: Body, world: World, prefix: str
    ) -> Connection:
        """
        Parses a given URDF joint and creates a corresponding connection object.

        The function processes the provided joint data, extracting necessary
        information including translation offsets, rotation offsets, connection type,
        and relevant joint limits. It maps URDF joint types to predefined connection
        types and either retrieves or creates a degree of freedom (DOF) in the world
        context. It generates and returns a connection object representing the
        relationship between a parent and a child body.

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

        lower_limits, upper_limits = urdf_joint_to_limits(joint)
        dof_name = connection_name
        multiplier = offset = None
        if joint.mimic:
            multiplier = (
                joint.mimic.multiplier if joint.mimic.multiplier is not None else 1
            )
            offset = joint.mimic.offset if joint.mimic.offset is not None else 0
            dof_name = PrefixedName(joint.mimic.joint, prefix)

        if dof_name not in [d.name for d in world.degrees_of_freedom]:
            dof = DegreeOfFreedom(
                name=dof_name,
                limits=DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits),
            )
            world.add_degree_of_freedom(dof)
        else:
            dof = world.get_degree_of_freedom_by_name(dof_name)

        assert joint.axis is not None, f"Joint axis is None for joint {joint.name}"
        result = connection_type(
            name=connection_name,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection,
            multiplier=multiplier,
            offset=offset,
            axis=Vector3(*map(int, joint.axis), reference_frame=parent),
            dof_id=dof.id,
        )
        return result

    def parse_link(self, link: urdfpy.Link, parent_frame: PrefixedName) -> Body:
        """
        Parses a URDF link to a link object.
        :param link: The URDF link to parse.
        :param parent_frame: The parent frame of the link, used for transformations of collisions and visuals.
        :return: The parsed link object.
        """
        name = PrefixedName(prefix=self.prefix, name=link.name)
        body = Body(name=name)
        visuals = self.parse_geometry(link.visuals, body)
        collisions = self.parse_geometry(link.collisions, body)
        body.visual = visuals
        body.collision = collisions
        return body

    def parse_geometry(
        self,
        geometry: Union[List[urdfpy.Collision], List[urdfpy.Visual]],
        body: Body,
    ) -> ShapeCollection:
        """
        Parses a URDF geometry to the corresponding shapes.
        :param geometry: The URDF geometry to parse either the collisions of visuals.'
        :param body: The body of the geometry, used for back referencing.
        :return: A List of shapes corresponding to the URDF geometry.
        """
        res = []
        material_dict = dict(
            zip(
                [material.name for material in self.parsed.materials],
                [
                    material.color.rgba if material.color else None
                    for material in self.parsed.materials
                ],
            )
        )
        for i, geom in enumerate(geometry):
            params = (*(geom.origin.xyz + geom.origin.rpy),) if geom.origin else ()
            origin_transform = HomogeneousTransformationMatrix.from_xyz_rpy(
                *params, reference_frame=body
            )
            if isinstance(geom.geometry, urdfpy.Box):
                color = (
                    Color(*material_dict.get(geom.material.name, (1, 1, 1, 1)))
                    if hasattr(geom, "material") and geom.material
                    else Color(1, 1, 1, 1)
                )
                res.append(
                    Box(
                        origin=origin_transform,
                        scale=Scale(*geom.geometry.size),
                        color=color,
                    )
                )
            elif isinstance(geom.geometry, urdfpy.Sphere):
                color = (
                    Color(*material_dict.get(geom.material.name, (1, 1, 1, 1)))
                    if hasattr(geom, "material") and geom.material
                    else Color(1, 1, 1, 1)
                )
                res.append(
                    Sphere(
                        origin=origin_transform,
                        radius=geom.geometry.radius,
                        color=color,
                    )
                )
            elif isinstance(geom.geometry, urdfpy.Cylinder):
                color = (
                    Color(*material_dict.get(geom.material.name, (1, 1, 1, 1)))
                    if hasattr(geom, "material") and geom.material
                    else Color(1, 1, 1, 1)
                )
                res.append(
                    Cylinder(
                        origin=origin_transform,
                        width=geom.geometry.radius,
                        height=geom.geometry.length,
                        color=color,
                    )
                )
            elif isinstance(geom.geometry, urdfpy.Mesh):
                if geom.geometry.filename is None:
                    raise ValueError("Mesh geometry must have a filename.")
                res.append(
                    FileMesh(
                        origin=origin_transform,
                        filename=self.parse_file_path(geom.geometry.filename),
                        scale=Scale(*(geom.geometry.scale or (1, 1, 1))),
                    )
                )
        return ShapeCollection(res, reference_frame=body)

    def parse_file_path(self, file_path: str) -> str:
        """
        Parses a file path which contains a ros package to a path in the local file system.

        :param file_path: The path to the URDF file.
        :return: The parsed and processed file path.
        """
        if "package://" in file_path:
            # Splits the file path at '//' to get the package  and the rest of the path
            package_split = file_path.split("//")
            # Splits the path after the // to get the package name and the rest of the path
            package_name = package_split[1].split("/")[0]
            try:
                from ament_index_python.packages import get_package_share_directory

                package_path = get_package_share_directory(package_name)
            except (ImportError, LookupError):
                if self.package_resolver:
                    if package_name in self.package_resolver:
                        package_path = self.package_resolver[package_name]
                    else:
                        raise ParsingError(
                            message=f"Package '{package_name}' not found in package resolver and "
                            f"ROS is not installed."
                        )
                else:
                    raise ParsingError(
                        message="No ROS install found while the URDF file contains references to "
                        "ROS packages."
                    )
            file_path = file_path.replace("package://" + package_name, package_path)
        if "file://" in file_path:
            file_path = file_path.replace("file://", "./")
        return file_path
