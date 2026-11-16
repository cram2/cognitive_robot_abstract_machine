from __future__ import annotations

from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.monitors.progress_monitors import (
    NoMonitoredTaskApproachingItsGoal,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition

from ...cramph_test.test_statechart.node_rebuild import RebuildLeavingNodesOut

# %% helpers


def _task_moving_the_bot(world: World, name: str) -> CartesianPosition:
    """
    :return: A task moving the cylinder bot to the origin of the world.
    """
    return CartesianPosition(
        name=name,
        root_link=world.root,
        tip_link=world.get_kinematic_structure_entity_by_name("bot"),
        goal_point=Point3(0, 0, 0, reference_frame=world.root),
    )


# %% building a node anew


def test_a_monitor_watching_tasks_watches_those_built_anew_that_are_not_left_out(
    cylinder_bot_world: World,
) -> None:
    kept = _task_moving_the_bot(cylinder_bot_world, "kept")
    left_out = _task_moving_the_bot(cylinder_bot_world, "left_out")
    rebuild = RebuildLeavingNodesOut(left_out=[left_out])

    rebuilt = rebuild.rebuilt(
        node=NoMonitoredTaskApproachingItsGoal(monitored_tasks=[kept, left_out])
    )

    assert rebuilt.monitored_tasks == [rebuild.built[kept]]


def test_a_monitor_watching_only_tasks_left_out_is_left_out(
    cylinder_bot_world: World,
) -> None:
    left_out = _task_moving_the_bot(cylinder_bot_world, "left_out")
    rebuild = RebuildLeavingNodesOut(left_out=[left_out])

    rebuilt = rebuild.rebuilt(
        node=NoMonitoredTaskApproachingItsGoal(monitored_tasks=[left_out])
    )

    assert rebuilt is None
