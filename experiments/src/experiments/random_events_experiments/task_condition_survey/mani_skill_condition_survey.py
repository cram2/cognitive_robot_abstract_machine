"""
This module reads the success condition every ManiSkill task states.

ManiSkill states a task's condition in ``evaluate``, which returns the outcome as the
``success`` entry of a mapping that also carries diagnostic measurements. Its conditions
are evaluated over a batch of environments at once, so they combine their parts with the
element-wise operators rather than with the boolean keywords.

The suite is read from source and never imported, so no simulator, no GPU and no
installation of ManiSkill's own dependencies is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import ClassVar

from experiments.random_events_experiments.task_condition_survey.condition_extraction import (
    ComputedConditionReader,
    ReturnedEntry,
    StatedOutcome,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    PredicateKind,
    PredicateNameRule,
    PredicateVocabulary,
)

# %% what ManiSkill calls its predicates


@dataclass
class ManiSkillPredicateVocabulary(PredicateVocabulary):
    """
    Classifies the predicates ManiSkill's success conditions apply.

    A predicate whose threshold the task itself writes bounds a continuous quantity,
    while one the task names without a threshold states a relation over a finite domain.
    So ``is_static(0.2)`` bounds a joint velocity and counts as continuous, whereas
    ``is_grasping(cube)`` names a contact relation and counts as discrete.

    ..note:: ``is_grasping`` is internally a bound on contact force and approach angle.
        Reading it as a relation is the same helper opacity the RoboCasa reading has, and
        is a limitation of classifying predicates at the level a task applies them.
    """

    exact_kinds: dict[str, PredicateKind] = field(
        default_factory=lambda: {
            "self.agent.is_static": PredicateKind.CONTINUOUS,
            "self.agent.is_grasping": PredicateKind.DISCRETE,
            "torch.abs": PredicateKind.CONTINUOUS,
            "abs": PredicateKind.CONTINUOUS,
            "torch.max": PredicateKind.CONTINUOUS,
            "torch.min": PredicateKind.CONTINUOUS,
            "torch.sum": PredicateKind.CONTINUOUS,
        }
    )
    """
    Kinds of the predicates the suite applies most often, keyed by dotted name.
    """

    name_rules: tuple[PredicateNameRule, ...] = (
        PredicateNameRule("norm", PredicateKind.CONTINUOUS),
        PredicateNameRule("dist", PredicateKind.CONTINUOUS),
        PredicateNameRule("angle", PredicateKind.CONTINUOUS),
        PredicateNameRule("contact_forces", PredicateKind.CONTINUOUS),
        PredicateNameRule("_thresh", PredicateKind.CONTINUOUS),
        PredicateNameRule("pose.p", PredicateKind.CONTINUOUS),
        PredicateNameRule("qpos", PredicateKind.CONTINUOUS),
        PredicateNameRule("qvel", PredicateKind.CONTINUOUS),
        PredicateNameRule("is_grasp", PredicateKind.DISCRETE),
        PredicateNameRule("is_static", PredicateKind.CONTINUOUS),
        PredicateNameRule("contact", PredicateKind.DISCRETE),
        PredicateNameRule("is_", PredicateKind.DISCRETE),
        PredicateNameRule("has_", PredicateKind.DISCRETE),
    )
    """
    Rules recognising the long tail of per-task predicates by naming convention,
    consulted in order once no exact name matched.
    """


# %% reading the suite


@dataclass
class ManiSkillConditionSurvey(ComputedConditionReader):
    """
    Reads every ManiSkill task's success condition.
    """

    vocabulary: PredicateVocabulary = field(
        default_factory=ManiSkillPredicateVocabulary
    )
    """
    Classifies the predicates ManiSkill's conditions apply.
    """

    condition_method_name: ClassVar[str] = "evaluate"
    """
    ManiSkill states a task's condition in ``evaluate``.
    """

    outcome: ClassVar[StatedOutcome] = ReturnedEntry(key="success")
    """
    ``evaluate`` returns the outcome as the ``success`` entry of a mapping.
    """

    @classmethod
    def for_installed_mani_skill(cls) -> ManiSkillConditionSurvey:
        """
        :return: A survey reading the task definitions of the installed ManiSkill.
        :raises SuiteSourceNotFound: When ManiSkill is not installed.
        """
        return cls(task_root=cls.installed_source_directory("mani_skill", "envs/tasks"))
