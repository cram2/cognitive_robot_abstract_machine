from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.pointing import Pointing, PointingCone
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from test.giskardpy_test.test_motion_statechart.debug_expression_helpers import (
    CURRENT_COLOR,
    GOAL_COLOR,
    debug_expression_by_name,
)


class TestPointingDebugExpressions:
    """
    Pointing tasks register the pointing axis (current) and goal axis (goal) as
    green/red debug expressions, named with the task name.
    """

    def test_pointing(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = Pointing(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            pointing_axis=Vector3(x=1, reference_frame=tip),
            name="point",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "point/goal")
        current = debug_expression_by_name(artifacts.debug_expressions, "point/current")
        assert isinstance(goal.expression, Vector3)
        assert isinstance(current.expression, Vector3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_pointing_cone(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = PointingCone(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            pointing_axis=Vector3(x=1, reference_frame=tip),
            cone_theta=0.1,
            name="cone",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "cone/goal")
        current = debug_expression_by_name(artifacts.debug_expressions, "cone/current")
        assert isinstance(goal.expression, Vector3)
        assert isinstance(current.expression, Vector3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR
