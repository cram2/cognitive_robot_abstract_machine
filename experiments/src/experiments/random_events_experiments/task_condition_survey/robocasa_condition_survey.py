"""
This module reads the success condition every RoboCasa task states.

RoboCasa states a task's condition in ``_check_success``, returning it directly, and
names its predicates after the object utilities the suite provides.

The suite is read from source and never imported, so neither RoboCasa's own version
assertions nor its assets are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import ClassVar

from experiments.random_events_experiments.task_condition_survey.condition_extraction import (
    ComputedConditionReader,
    ReturnedValue,
    StatedOutcome,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    PredicateKind,
    PredicateNameRule,
    PredicateVocabulary,
)

# %% what RoboCasa calls its predicates


@dataclass
class RoboCasaPredicateVocabulary(PredicateVocabulary):
    """
    Classifies the predicates RoboCasa's success conditions apply.

    Recognises the frequent predicates by their exact name and the long tail by naming
    convention, so a task using a per-fixture check the survey has never seen is still
    classified rather than silently dropped.
    """

    exact_kinds: dict[str, PredicateKind] = field(
        default_factory=lambda: {
            "OU.gripper_obj_far": PredicateKind.CONTINUOUS,
            "OU.gripper_fxtr_far": PredicateKind.CONTINUOUS,
            "OU.obj_fixture_bbox_min_dist": PredicateKind.CONTINUOUS,
            "self.dist_between_obj": PredicateKind.CONTINUOUS,
            "OU.check_obj_in_receptacle": PredicateKind.DISCRETE,
            "OU.obj_inside_of": PredicateKind.DISCRETE,
            "OU.check_obj_fixture_contact": PredicateKind.DISCRETE,
            "OU.check_obj_any_counter_contact": PredicateKind.DISCRETE,
            "self.check_contact": PredicateKind.DISCRETE,
            "abs": PredicateKind.CONTINUOUS,
            "np.abs": PredicateKind.CONTINUOUS,
            "min": PredicateKind.CONTINUOUS,
            "max": PredicateKind.CONTINUOUS,
            "isinstance": PredicateKind.DISCRETE,
            "hasattr": PredicateKind.DISCRETE,
        }
    )
    """
    Kinds of the predicates the suite applies most often, keyed by dotted name.
    """

    name_rules: tuple[PredicateNameRule, ...] = (
        PredicateNameRule("_far", PredicateKind.CONTINUOUS),
        PredicateNameRule("_dist", PredicateKind.CONTINUOUS),
        PredicateNameRule("norm", PredicateKind.CONTINUOUS),
        PredicateNameRule("close_to", PredicateKind.CONTINUOUS),
        PredicateNameRule("check_", PredicateKind.DISCRETE),
        PredicateNameRule("is_", PredicateKind.DISCRETE),
        PredicateNameRule("has_", PredicateKind.DISCRETE),
        PredicateNameRule("get_state", PredicateKind.DISCRETE),
        PredicateNameRule("_state", PredicateKind.DISCRETE),
        PredicateNameRule("inside_of", PredicateKind.DISCRETE),
        PredicateNameRule("_on_", PredicateKind.DISCRETE),
    )
    """
    Rules recognising the long tail of per-fixture predicates by naming convention,
    consulted in order once no exact name matched.
    """


# %% reading the suite


@dataclass
class RoboCasaConditionSurvey(ComputedConditionReader):
    """
    Reads every RoboCasa task's success condition.
    """

    vocabulary: PredicateVocabulary = field(default_factory=RoboCasaPredicateVocabulary)
    """
    Classifies the predicates RoboCasa's conditions apply.
    """

    condition_method_name: ClassVar[str] = "_check_success"
    """
    RoboCasa states a task's condition in ``_check_success``.
    """

    outcome: ClassVar[StatedOutcome] = ReturnedValue()
    """
    ``_check_success`` returns the outcome itself.
    """

    @classmethod
    def for_installed_robocasa(cls) -> RoboCasaConditionSurvey:
        """
        :return: A survey reading the task definitions of the installed RoboCasa.
        :raises SuiteSourceNotFound: When RoboCasa is not installed.
        """
        return cls(
            task_root=cls.installed_source_directory("robocasa", "environments/kitchen")
        )
