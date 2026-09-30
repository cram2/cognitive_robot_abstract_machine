from pathlib import Path

import pytest

from coraplex.datastructures.enums import Arms
from cramera.generated_json import write_json_atomically
from cramera.knowledge.enums import KinematicChainGroup
from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.scene_bundle import SceneBundle
from cramera.knowledge.views.dispatcher import GraphPanelViews
from cramera.robot_parts import ArmSide, RobotPartAnnotation, RobotPartRole
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2LeftArm, PR2LeftGripper
from semantic_digital_twin.world import World


# %% recorded annotation types
def test_recorded_part_side_uses_the_native_arms_enum() -> None:
    annotation = RobotPartAnnotation(
        name="Manipulator", role=RobotPartRole.ARM, side=ArmSide.LEFT
    )

    restored = RobotPartAnnotation.from_payload(annotation.to_payload())

    assert restored.side is Arms.LEFT


@pytest.mark.parametrize(
    ("name", "role", "side", "group"),
    [
        (
            "right_gripper",
            RobotPartRole.ARM,
            ArmSide.LEFT,
            KinematicChainGroup.LEFT_ARM,
        ),
        ("left_arm", RobotPartRole.ARM, ArmSide.RIGHT, KinematicChainGroup.RIGHT_ARM),
        (
            "left_arm",
            RobotPartRole.END_EFFECTOR,
            ArmSide.LEFT,
            KinematicChainGroup.GRIPPER,
        ),
        ("left_arm", RobotPartRole.ARM, None, KinematicChainGroup.BASE),
    ],
)
def test_kinematics_uses_recorded_role_and_side(
    fixture_scene: Path,
    name: str,
    role: RobotPartRole,
    side: ArmSide | None,
    group: KinematicChainGroup,
) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["parts"] = {name: ["l_shoulder_link"]}
    scene["robot"]["partAnnotations"] = [
        RobotPartAnnotation(name, role, side, ["l_shoulder_link"]).to_payload()
    ]
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert groups["urdf:l_shoulder_link"] is group


def test_kinematics_reads_links_without_the_legacy_part_map(
    fixture_scene: Path,
) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["parts"] = {}
    scene["robot"]["partAnnotations"] = [
        RobotPartAnnotation(
            "Manipulator", RobotPartRole.ARM, ArmSide.RIGHT, ["l_shoulder_link"]
        ).to_payload()
    ]
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert groups["urdf:l_shoulder_link"] is KinematicChainGroup.RIGHT_ARM


def test_unknown_legacy_parts_are_not_classified_by_substrings(
    fixture_scene: Path,
) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["parts"] = {"left_handrail": ["l_shoulder_link"]}
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert groups["urdf:l_shoulder_link"] is KinematicChainGroup.BASE


def test_recorded_arm_queries_use_the_native_arms_enum(fixture_scene: Path) -> None:
    query = (Path(__file__).parent / "dataset" / "recorded_left_arm.eql").read_text()

    result = EqlSession.of_active_scene().run(query)

    assert result.ok
    assert result.count == 1
    assert result.rows[0]["__entity__"] == "left_arm"


def test_recorded_arm_type_preserves_the_saved_query_name(fixture_scene: Path) -> None:
    session = EqlSession.of_active_scene()
    namespace = session.namespace()

    assert namespace["Arm"] is namespace["RecordedArm"]
    assert type(session.knowledge_base.arms[0]) is namespace["RecordedArm"]


def test_gripper_membership_takes_precedence_over_its_arm(fixture_scene: Path) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["partAnnotations"] = [
        RobotPartAnnotation(
            "Tool",
            RobotPartRole.END_EFFECTOR,
            Arms.LEFT,
            ["l_gripper_link"],
            "Manipulator",
        ).to_payload(),
        RobotPartAnnotation(
            "Manipulator",
            RobotPartRole.ARM,
            Arms.LEFT,
            ["l_shoulder_link", "l_gripper_link"],
        ).to_payload(),
    ]
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert groups["urdf:l_gripper_link"] is KinematicChainGroup.GRIPPER


def test_legacy_center_arm_retains_its_gripper(fixture_scene: Path) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["parts"] = {
        "center_arm": ["l_shoulder_link"],
        "gripper": ["l_gripper_link"],
    }
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    [arm] = EqlSession.of_active_scene().knowledge_base.arms

    assert arm.side is None
    assert arm.gripper.name == "gripper"


def test_recorded_sensor_role_overrides_a_misleading_name(fixture_scene: Path) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["partAnnotations"] = [
        {
            "name": "left_arm",
            "role": "sensor",
            "side": None,
            "links": ["l_shoulder_link"],
            "attachedTo": None,
        }
    ]
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert groups["urdf:l_shoulder_link"] is KinematicChainGroup.SENSOR


def test_native_sensors_are_recorded_as_sensor_parts(pr2_world_copy: World) -> None:
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(AbstractRobot)
    annotations = RobotPartAnnotation.of_robot(robot)

    sensor_parts = {
        part.name: part.links for part in annotations if part.role.value == "sensor"
    }
    expected = {
        type(sensor).__name__: sorted(set(RobotPartAnnotation.link_names(sensor)))
        for sensor in robot.get_sensors()
    }

    assert expected
    assert sensor_parts == expected


def test_legacy_native_part_names_keep_their_recorded_links(
    fixture_scene: Path,
) -> None:
    scene = SceneBundle.of_active_scene().scene
    scene["robot"]["parts"] = {
        PR2LeftArm.__name__: ["l_shoulder_link"],
        PR2LeftGripper.__name__: ["l_gripper_link"],
    }
    write_json_atomically(fixture_scene / "scenes" / "fixture" / "scene.json", scene)

    [arm] = EqlSession.of_active_scene().knowledge_base.arms
    payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
    groups = {node.id: node.group for node in payload.nodes}

    assert arm.name == PR2LeftArm.__name__
    assert arm.gripper.name == PR2LeftGripper.__name__
    assert groups["urdf:l_shoulder_link"] is KinematicChainGroup.LEFT_ARM
    assert groups["urdf:l_gripper_link"] is KinematicChainGroup.GRIPPER
