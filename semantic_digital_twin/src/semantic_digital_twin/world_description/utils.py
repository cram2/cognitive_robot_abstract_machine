from typing_extensions import Type

from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DiffDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body


def world_with_urdf_factory(
    urdf_path: str,
    robot_semantic_annotation: Type[AbstractRobot] | None,
    drive_connection_type: Type[OmniDrive | DiffDrive],
    robot_starting_pose: HomogeneousTransformationMatrix | None = None,
    urdf_path_resolver: PathResolver | None = None,
):
    """
    Builds this tree:
    map -> odom_combined -> "urdf tree"
    """
    urdf_parser = URDFParser.from_file(
        file_path=urdf_path, path_resolver=urdf_path_resolver
    )
    world_with_urdf = urdf_parser.parse()
    if robot_semantic_annotation is not None:
        robot_semantic_annotation.from_world(world_with_urdf)

    with world_with_urdf.modify_world():
        map = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))

        map_C_localization = Connection6DoF.create_with_dofs(
            world_with_urdf, map, localization_body
        )
        world_with_urdf.add_connection(map_C_localization)

        c_root_bf = drive_connection_type.create_with_dofs(
            parent=localization_body,
            child=world_with_urdf.root,
            world=world_with_urdf,
        )
        world_with_urdf.add_connection(c_root_bf)
        c_root_bf.has_hardware_interface = True
        if robot_starting_pose is not None:
            c_root_bf.origin = robot_starting_pose

    return world_with_urdf
