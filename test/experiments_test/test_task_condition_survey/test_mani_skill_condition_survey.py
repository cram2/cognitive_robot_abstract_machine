"""
Reading the success condition a ManiSkill task states.

ManiSkill phrases its conditions differently from a suite that returns a truth value:
the outcome is one entry of a returned mapping, and its parts are combined element-wise
because a condition is evaluated over a batch of environments at once. Each test here
pins one of those phrasings, or one classification the suite's own vocabulary makes.
"""

import pathlib
import textwrap

from experiments.random_events_experiments.task_condition_survey.mani_skill_condition_survey import (
    ManiSkillConditionSurvey,
    ManiSkillPredicateVocabulary,
)
from experiments.random_events_experiments.task_condition_survey.set_kind_survey import (
    ManiSkillPredicateGeometry,
    PredicateGeometry,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConditionKind,
    PredicateKind,
    StatedTaskCondition,
)


def read(source: str, task_root: pathlib.Path) -> dict[str, StatedTaskCondition]:
    """
    :param source: A module defining task classes, as written in the suite.
    :param task_root: Directory to write it into and read it back from.
    :return: The condition each task class in it states.
    """
    (task_root / "tasks.py").write_text(textwrap.dedent(source))
    return ManiSkillConditionSurvey(task_root=task_root).conditions()


# %% how the suite states an outcome


def test_outcome_is_read_from_the_success_entry_of_the_returned_mapping(
    tmp_path: pathlib.Path,
):
    """
    A task reports diagnostic measurements alongside its outcome, so only the entry
    naming the outcome states the condition.
    """
    conditions = read(
        """
        class PickCube:
            def evaluate(self):
                is_placed = torch.linalg.norm(self.goal.p - self.cube.p, axis=1) <= 0.05
                is_grasped = self.agent.is_grasping(self.cube)
                return {"success": is_placed, "is_grasped": is_grasped}
        """,
        tmp_path,
    )

    [condition] = conditions.values()
    assert [proposition.stated_under for proposition in condition.propositions()] == [
        "torch.linalg.norm"
    ]


def test_condition_combining_a_distance_and_a_relation_is_hybrid(
    tmp_path: pathlib.Path,
):
    """
    A task requiring both a bounded distance and a held object states a condition of
    both kinds at once, which is the kind of set the survey counts.
    """
    conditions = read(
        """
        class StackCube:
            def evaluate(self):
                is_placed = torch.linalg.norm(self.goal.p - self.cube.p, axis=1) <= 0.05
                is_grasped = self.agent.is_grasping(self.cube)
                return dict(success=is_placed & ~is_grasped)
        """,
        tmp_path,
    )

    [condition] = conditions.values()
    assert condition.kind is ConditionKind.HYBRID


def test_task_without_a_condition_method_is_not_reported(tmp_path: pathlib.Path):
    """
    A class that never states a success condition is not a task the survey can read, so
    it is left out rather than reported as stating nothing.
    """
    conditions = read(
        """
        class NotATask:
            def reset(self):
                return None
        """,
        tmp_path,
    )

    assert conditions == {}


# %% what the suite calls its predicates


def test_threshold_written_by_the_task_bounds_a_continuous_quantity():
    """
    A predicate the task applies with its own threshold bounds a quantity, however the
    predicate is named.
    """
    vocabulary = ManiSkillPredicateVocabulary()

    assert vocabulary.classify("self.agent.is_static") is PredicateKind.CONTINUOUS


def test_relation_named_without_a_threshold_states_a_finite_domain():
    """
    A predicate naming a relation the task does not put a threshold on states a
    condition over a finite domain, which is what makes a condition combining it hybrid.
    """
    vocabulary = ManiSkillPredicateVocabulary()

    assert vocabulary.classify("self.agent.is_grasping") is PredicateKind.DISCRETE


def test_position_compared_against_another_quantity_is_continuous():
    """
    A position or joint coordinate is a continuous quantity whether or not the value it
    is compared against is written as a literal.
    """
    vocabulary = ManiSkillPredicateVocabulary()

    assert vocabulary.classify("self.box.pose.p") is PredicateKind.CONTINUOUS
    assert vocabulary.classify("self.valve.qpos") is PredicateKind.CONTINUOUS


# %% what shape those predicates describe
#
# The shapes are read through the reader rather than from hand-built propositions: what a
# measure was applied to is only recorded when a condition is actually read, so a test that
# supplies that text itself cannot fail when the reader stops producing it.


def shape_of(source: str, task_root: pathlib.Path) -> PredicateGeometry:
    """
    :param source: A module defining one task, as written in the suite.
    :param task_root: Directory to write it into and read it back from.
    :return: The shape ManiSkill's vocabulary assigns the condition's single proposition.
    """
    [condition] = read(source, task_root).values()
    [proposition] = condition.propositions()
    return ManiSkillPredicateGeometry().geometry_of(proposition)


def test_distance_taken_over_a_whole_position_is_a_ball(tmp_path: pathlib.Path):
    """
    A distance between two whole positions is taken in three dimensions, so it describes
    a ball.
    """
    shape = shape_of(
        """
        class PickCube:
            def evaluate(self):
                placed = torch.linalg.norm(self.goal.p - self.cube.p, axis=1) <= 0.05
                return {"success": placed}
        """,
        tmp_path,
    )

    assert shape is PredicateGeometry.EUCLIDEAN_BALL


def test_distance_taken_over_a_sliced_position_is_a_disc(tmp_path: pathlib.Path):
    """
    A distance taken after slicing a position to its first two axes is a distance in the
    plane, so it describes a disc rather than a ball -- which means the reader has to
    record what the measure was applied to and not only what it is called.
    """
    shape = shape_of(
        """
        class StackCube:
            def evaluate(self):
                offset = self.cubeA.pose.p - self.cubeB.pose.p
                aligned = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
                return {"success": aligned}
        """,
        tmp_path,
    )

    assert shape is PredicateGeometry.EUCLIDEAN_DISC


def test_velocity_bound_the_task_writes_is_a_bound_per_axis(tmp_path: pathlib.Path):
    """
    Requiring the robot to be still bounds each joint's velocity separately, so the
    region is already a box and its representation loses nothing.
    """
    shape = shape_of(
        """
        class PlaceCube:
            def evaluate(self):
                return {"success": self.agent.is_static(0.2)}
        """,
        tmp_path,
    )

    assert shape is PredicateGeometry.AXIS_ALIGNED
