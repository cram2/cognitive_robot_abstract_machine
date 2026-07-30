import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MismatchedTrajectoryLengthsError
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body

# %% builders


def build_connection(parent_name: str, child_name: str) -> FixedConnection:
    """
    Build a standalone connection usable as a trajectory key.
    """
    return FixedConnection(
        parent=Body(name=PrefixedName(parent_name)),
        child=Body(name=PrefixedName(child_name)),
    )


# %% is_empty


def test_is_empty_without_tracked_connections():
    assert MotionTrajectory().is_empty() is True


def test_is_empty_with_tracked_connection_but_no_positions():
    trajectory = MotionTrajectory(data={build_connection("a", "b"): []})
    assert trajectory.is_empty() is True


def test_is_not_empty_with_recorded_positions():
    trajectory = MotionTrajectory(data={build_connection("a", "b"): [0.1]})
    assert trajectory.is_empty() is False


# %% position_updates_at


def test_position_updates_at_returns_every_connection_position_at_step():
    first_connection = build_connection("a", "b")
    second_connection = build_connection("c", "d")
    trajectory = MotionTrajectory(
        data={first_connection: [0.1, 0.2], second_connection: [1.0, 2.0]}
    )
    assert trajectory.position_updates_at(1) == {
        first_connection: 0.2,
        second_connection: 2.0,
    }


# %% positions_for


def test_positions_for_tracked_connection():
    connection = build_connection("a", "b")
    trajectory = MotionTrajectory(data={connection: [0.1, 0.2, 0.3]})
    assert trajectory.positions_for(connection) == [0.1, 0.2, 0.3]


def test_positions_for_untracked_connection_is_empty():
    tracked_connection = build_connection("a", "b")
    untracked_connection = build_connection("c", "d")
    trajectory = MotionTrajectory(data={tracked_connection: [0.1, 0.2]})
    assert trajectory.positions_for(untracked_connection) == []


# %% lock-step validation


def test_mismatched_sequence_lengths_are_rejected():
    first_connection = build_connection("a", "b")
    second_connection = build_connection("c", "d")
    with pytest.raises(MismatchedTrajectoryLengthsError) as error_info:
        MotionTrajectory(data={first_connection: [0.1, 0.2], second_connection: [1.0]})
    assert error_info.value.lengths_by_connection == {
        first_connection.name: 2,
        second_connection.name: 1,
    }
