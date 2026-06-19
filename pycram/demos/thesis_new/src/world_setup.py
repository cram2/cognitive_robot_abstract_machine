import os
from typing import Type

from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    OmniDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body


def world_with_urdf_factory(
    urdf_path: str,
    robot_semantic_annotation: Type[AbstractRobot] | None,
    drive_connection_type: Type[OmniDrive | DifferentialDrive],
    robot_starting_pose: HomogeneousTransformationMatrix | None = None,
    urdf_path_resolver: PathResolver | None = None,
    robot_localization_pose: HomogeneousTransformationMatrix | None = None,
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
    if robot_localization_pose is not None:
        map_C_localization.origin = robot_localization_pose

    if robot_starting_pose is not None:
        c_root_bf.origin = robot_starting_pose

    return world_with_urdf


THESIS_NEW_DEFAULT_ROBOT = "pr2"
THESIS_NEW_ROBOT_ENV = "THESIS_NEW_ROBOT"
THESIS_NEW_DEFAULT_ENVIRONMENT = "apartment"
THESIS_NEW_ENVIRONMENT_ENV = "THESIS_NEW_ENVIRONMENT"
DEFAULT_ENVIRONMENT_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(
    1.5, 2.0, 0.0
)
KITCHEN_LIKE_ENVIRONMENTS = {"kitchen", "test-kitchen-chat"}
ENVIRONMENT_START_POSES = {
    "apartment": HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.0, 0.0),
    "apartment_without_walls": HomogeneousTransformationMatrix.from_xyz_rpy(
        1.5, 2.0, 0.0
    ),
    "kitchen": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
    "test-kitchen-chat": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
    "robocup": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
    "suturo": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
    "isr": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
    "isr-testbed": HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0.0),
}
RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "resources")
)
WORLDS_DIR = os.path.join(RESOURCES_DIR, "worlds")
EXTERNAL_ENVIRONMENT_SPECS = {
    "isr": "package://isr_testbed/urdf/isr-testbed.urdf",
}
# robot_urdf, robot_cls, drive_cls, robot_off
ROBOT_SPECS = {
    "pr2": (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro",
        PR2,
        OmniDrive,
        HomogeneousTransformationMatrix(),
    ),
    "hsrb": (
        os.path.join(RESOURCES_DIR, "robots", "hsrb.urdf"),
        HSRB,
        OmniDrive,
        HomogeneousTransformationMatrix(),
    ),
    "stretch": (
        os.path.join(RESOURCES_DIR, "robots", "stretch_description.urdf"),
        Stretch,
        DifferentialDrive,
        HomogeneousTransformationMatrix(),
    ),
    "tiago": (
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        Tiago,
        DifferentialDrive,
        HomogeneousTransformationMatrix(),
    ),
    "armar7": (
        "package://iai_kit_armar7/urdf/Armar7.urdf",
        Armar7,
        OmniDrive,
        HomogeneousTransformationMatrix(),
    ),
    "garmi": (
        "package://garmi_description/urdf/garmi.urdf",
        Garmi,
        OmniDrive,
        HomogeneousTransformationMatrix(),
    ),
    "justin": (
        "package://iai_dlr_rollin_justin/urdf/rollin_justin.urdf",
        Justin,
        OmniDrive,
        HomogeneousTransformationMatrix(),
    ),
    "g1": (
        "package://iai_offis_g1_description/urdf/offis_unitree_g1.urdf",
        UnitreeG1,
        OmniDrive,
        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.8),
    ),
    # "donbot": (
    #     "package://iai_donbot_description//urdf/offis_unitree_g1.urdf",
    #     UnitreeG1,
    #     OmniDrive,
    #     HomogeneousTransformationMatrix.from_xyz_rpy(z=0.8),
    # ),
}
ROBOT_NAME_ALIASES = {
    "rollin_justin": "justin",
    "unitree_g1": "g1",
}


def resolve_robot_name(robot_name=None):
    selected = robot_name or os.environ.get(
        THESIS_NEW_ROBOT_ENV, THESIS_NEW_DEFAULT_ROBOT
    )
    normalized = str(selected).strip().lower()
    normalized = ROBOT_NAME_ALIASES.get(normalized, normalized)
    if normalized not in ROBOT_SPECS:
        supported = ", ".join(sorted([*ROBOT_SPECS, *ROBOT_NAME_ALIASES]))
        raise ValueError(
            f"Unsupported thesis_new robot '{selected}'. Supported: {supported}"
        )
    return normalized


def resolve_robot_name_from_annotation(robot):
    for robot_name, (_, robot_cls, _, _) in ROBOT_SPECS.items():
        if isinstance(robot, robot_cls):
            return robot_name
    raise ValueError(
        f"Unsupported semantic robot annotation type: {type(robot).__name__}"
    )


def _supported_environment_names():
    print("using external environments")
    supported = list(EXTERNAL_ENVIRONMENT_SPECS)
    for filename in os.listdir(WORLDS_DIR):
        file_path = os.path.join(WORLDS_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith(".urdf"):
            supported.append(os.path.splitext(filename)[0])
    return sorted(supported)


def resolve_environment_name(environment_name=None):
    selected = environment_name or os.environ.get(
        THESIS_NEW_ENVIRONMENT_ENV, THESIS_NEW_DEFAULT_ENVIRONMENT
    )
    normalized = str(selected).strip()

    if normalized.startswith("package://") or os.path.isabs(normalized):
        return os.path.splitext(os.path.basename(normalized))[0]

    return normalized[:-5] if normalized.endswith(".urdf") else normalized


def resolve_environment_path(environment_name=None):
    selected = environment_name or os.environ.get(
        THESIS_NEW_ENVIRONMENT_ENV, THESIS_NEW_DEFAULT_ENVIRONMENT
    )

    if isinstance(selected, str) and (
        selected.startswith("package://") or os.path.isabs(selected)
    ):
        return selected

    normalized = resolve_environment_name(selected)

    if normalized in EXTERNAL_ENVIRONMENT_SPECS:
        return EXTERNAL_ENVIRONMENT_SPECS[normalized]

    environment_path = os.path.join(WORLDS_DIR, f"{normalized}.urdf")
    if os.path.isfile(environment_path):
        return environment_path

    supported = ", ".join(_supported_environment_names())
    raise ValueError(
        f"Unsupported thesis_new environment '{selected}'. Supported: {supported}"
    )


def resolve_environment_start_pose(environment_name=None):
    normalized = resolve_environment_name(environment_name)
    return ENVIRONMENT_START_POSES.get(normalized, DEFAULT_ENVIRONMENT_START_POSE)


def setup_thesis_world(robot_name=None, environment_name=None):
    resolved_robot_name = resolve_robot_name(robot_name)
    robot_urdf, robot_cls, drive_cls, robot_off = ROBOT_SPECS[resolved_robot_name]

    robot_start_pose = resolve_environment_start_pose(environment_name)
    environment_path = resolve_environment_path(environment_name)

    environment_world = URDFParser.from_file(environment_path).parse()

    # pipeline = Pipeline(steps=[
    #     CenterLocalGeometryAndPreserveWorldPose()
    # ])
    # environment_world = pipeline.apply(world=environment_world)

    robot_world = world_with_urdf_factory(
        urdf_path=robot_urdf,
        robot_semantic_annotation=robot_cls,
        drive_connection_type=drive_cls,
        robot_localization_pose=robot_off,
        robot_starting_pose=robot_start_pose,
    )
    robot_world.merge_world(environment_world)
    # environment_world.merge_world_at_pose(robot_world, robot_start_pose)
    # robot_cls.from_world(environment_world)

    print(robot_world.root)
    # print(robot_world.bodies)

    from semantic_digital_twin.spatial_computations.raytracer import RayTracer

    #
    # rt = RayTracer(robot_world)
    # rt.scene.show()
    # viewer = rt.scene.show("jupyter")
    # from IPython.display import display
    #
    # display(rt.scene.show("jupyter"))
    #
    # print("unter rt scene")
    return robot_world
