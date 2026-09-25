"""
Tests for watching an executing motion statechart.
"""

from .test_live_bridge import make_chart

from giskardpy.motion_statechart.data_types import TransitionKind
from giskardpy.motion_statechart.graph_node import Goal, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from cramera.knowledge.recorded_statecharts import RecordedStatecharts
from cramera.live.chart_observer import ChartObserver
from cramera.live.chart_structure import ObservationName, structure_of

# %% what one tick of a statechart looks like


class TestSnapshotOfAnExecutingChart:
    """
    A statechart exists only while it is being ticked, so what a viewer shows of it --
    or what a recording keeps of it -- is a snapshot taken per tick.
    """

    def test_nothing_is_executing_before_a_chart_is_seen(self):
        assert ChartObserver().snapshot(None) is None

    def test_a_chart_is_snapshotted_with_its_nodes(self):
        snapshot = ChartObserver().snapshot(make_chart())

        assert [node.name for node in snapshot.nodes] == [
            "Goal",
            "MoveJoints",
            "JointGoalReached",
        ]

    def test_the_transitions_are_snapshotted_too(self):
        snapshot = ChartObserver().snapshot(make_chart())

        assert [edge.kind for edge in snapshot.edges] == ["START", "END"]

    def test_every_node_says_where_it_stands(self):
        snapshot = ChartObserver().snapshot(make_chart())

        assert [node.observation for node in snapshot.nodes] == [
            ObservationName.UNKNOWN,
            ObservationName.UNKNOWN,
            ObservationName.FALSE,
        ]

    def test_a_chart_that_has_not_moved_is_still_snapshotted(self):
        observer = ChartObserver()
        chart = make_chart()
        observer.snapshot(chart)

        assert observer.snapshot(chart) is not None

    def test_a_chart_whose_nodes_moved_is_snapshotted_again(self):
        observer = ChartObserver()
        observer.snapshot(make_chart())

        assert observer.snapshot(make_chart(life_cycle=(1, 2, 0))) is not None

    def test_the_title_travels_into_the_snapshot(self):
        snapshot = ChartObserver(title="PickUpAction").snapshot(make_chart())

        assert snapshot.title == "PickUpAction"


# %% what the live wire sends


class TestOnlyChangesGoOnTheWire:
    """
    A live viewer already holds the last chart it was sent, so re-sending an unchanged
    one is wasted traffic.

    A recording cannot dedupe that way: a tick with nothing to say
    there means the chart stopped, not that it stood still.
    """

    def test_the_first_look_is_a_change(self):
        assert ChartObserver().change(make_chart()) is not None

    def test_a_chart_that_has_not_moved_is_not_sent_twice(self):
        observer = ChartObserver()
        chart = make_chart()
        observer.change(chart)

        assert observer.change(chart) is None

    def test_a_chart_whose_nodes_moved_is_sent_again(self):
        observer = ChartObserver()
        observer.change(make_chart())

        assert observer.change(make_chart(life_cycle=(1, 2, 0))) is not None

    def test_nothing_executing_is_not_a_change(self):
        assert ChartObserver().change(None) is None


# %% structural identity
def test_transition_kind_changes_the_chart_signature() -> None:
    """
    A changed transition requires rebuilding the graph despite stable node names.
    """
    original = make_chart()
    changed = make_chart()
    changed.rx_graph.edges[0][2].kind = TransitionKind.END

    assert structure_of(original).signature != structure_of(changed).signature


def test_transition_direction_changes_the_chart_signature() -> None:
    """
    A rewired edge cannot retain the previous graph's structural identity.
    """
    original = make_chart()
    changed = make_chart()
    source, target, transition = changed.rx_graph.edges[0]
    changed.rx_graph.edges[0] = (target, source, transition)

    assert structure_of(original).signature != structure_of(changed).signature


def test_parent_changes_the_chart_signature() -> None:
    """
    Reparenting a named node changes its graph hierarchy.
    """
    original = make_chart()
    changed = make_chart()
    changed.nodes[1].parent_node_index = None

    assert structure_of(original).signature != structure_of(changed).signature


def test_node_type_changes_the_chart_signature() -> None:
    """
    Replacing a named task with a goal changes how the graph renders it.
    """
    original = MotionStatechart()
    original.add_node(Task())
    changed = MotionStatechart()
    changed.add_node(Goal(name=original.nodes[0].name))

    assert structure_of(original).signature != structure_of(changed).signature


def test_recording_preserves_rewired_chart_structures() -> None:
    """
    A later chart's transitions survive recording when its node names are reused.
    """
    original = make_chart()
    changed = make_chart()
    changed.rx_graph.edges[0][2].kind = TransitionKind.END
    observer = ChartObserver()
    snapshots = [observer.snapshot(original), observer.snapshot(changed)]

    recorded = RecordedStatecharts.of_snapshots(snapshots)

    assert len(recorded.charts) == len(snapshots)
    assert [chart.edges[0].kind for chart in recorded.charts] == [
        snapshot.edges[0].kind for snapshot in snapshots
    ]
