from semantic_digital_twin.robots.xarm5 import XArm5


def test_xarm5_loads_from_workspace_description(xarm5_world):
    """
    The xArm5 description comes from the ``xarm_description`` package of the
    workspace (see ``.github/docker/setup_workspace.py``); its device xacro
    must be expanded with :meth:`XArm5.get_xacro_mappings` because the xArm 5
    is not the default model of that parameterized description.
    """
    robot = xarm5_world.get_semantic_annotations_by_type(XArm5)[0]
    assert robot.root.name.name == "link_base"
    assert robot.arm.tip.name.name == "link5"
    assert robot.arm.end_effector.root.name.name == "link_eef"
