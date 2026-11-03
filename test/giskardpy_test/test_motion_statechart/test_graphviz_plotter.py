from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from cramph.node import CancelStatechart
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    Task,
)
from cramph.plotters.styles import (
    NodeDrawingStyle,
)

# %% plot specifications of motion nodes


def test_collision_avoidance_goals_collapse_their_children():
    assert ExternalCollisionAvoidance().plot_specifications.collapse_children
    assert SelfCollisionAvoidance().plot_specifications.collapse_children


def test_terminal_nodes_and_tasks_have_correct_plot_specifications():
    assert EndMotion().plot_specifications.extra_border_styles == ["rounded"]
    assert CancelStatechart(
        exception=Exception()
    ).plot_specifications.extra_border_styles == ["dashed, rounded"]
    assert Task().plot_specifications.style == NodeDrawingStyle.TASK.style
    assert Task().plot_specifications.shape == NodeDrawingStyle.TASK.shape
