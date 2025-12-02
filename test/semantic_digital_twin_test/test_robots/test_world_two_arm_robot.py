from semantic_digital_twin.testing import two_arm_robot_world
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def test_simple_two_arm_robot(two_arm_robot_world: World):
    for c in two_arm_robot_world.connections:
        assert isinstance(c.origin_expression.reference_frame, Body), c.name
        assert isinstance(c.origin_expression.child_frame, Body), c.name
    for b in two_arm_robot_world.bodies:
        for c in b.collision:
            assert isinstance(c.origin.reference_frame, Body), b.name
            assert c.origin.child_frame is None, b.name
        for v in b.visual:
            assert isinstance(v.origin.reference_frame, Body), b.name
            assert v.origin.child_frame is None, b.name
