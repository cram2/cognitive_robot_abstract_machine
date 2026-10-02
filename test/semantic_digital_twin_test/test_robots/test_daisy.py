from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_specification import (
    GriplinkFlexSpecification,
    GriplinkPresetSpecification,
)
from semantic_digital_twin.robots.daisy import DAiSy


def test_daisy_grippers_build_griplink_specifications(daisy_world):
    """
    Both DAiSy end effectors build griplink-specific specifications from their declared
    states, so generic actions produce robot-appropriate specifications without naming
    the robot.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.get_end_effectors():
        open_spec = end_effector.default_specification(GripperState.OPEN)
        close_spec = end_effector.default_specification(GripperState.CLOSE)
        flex_close_spec = end_effector.default_specification(GripperState.FLEXCLOSE)

        assert isinstance(open_spec, GriplinkPresetSpecification)
        assert isinstance(close_spec, GriplinkPresetSpecification)
        assert isinstance(flex_close_spec, GriplinkFlexSpecification)
