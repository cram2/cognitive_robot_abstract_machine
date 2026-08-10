"""
Surveying which kinds of set a dataset's conditions describe, how often, and what
representing them costs and captures.

The conditions here are built directly rather than parsed, so each test pins one
property of the survey -- how a kind is decided, how a shape is assigned, how a
frequency is counted -- independently of how any task suite happens to phrase its
conditions.
"""

import math

from krrood.entity_query_language.factories import and_, exists, for_all, or_

from experiments.random_events_experiments.task_condition_survey.set_kind_survey import (
    GeometryMarker,
    PredicateGeometry,
    PredicateGeometryVocabulary,
    RoboCasaPredicateGeometry,
    SetKindSurvey,
    SurveyedDataset,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConditionKind,
    PredicateKind,
    SceneObject,
    StatedTaskCondition,
)

from .test_task_conditions import condition, relation


def proposition(name: str, kind: PredicateKind = PredicateKind.DISCRETE):
    """
    :param name: Name the relation is stated under.
    :param kind: What kind of set it describes.
    :return: The proposition that relation states.
    """
    [stated] = condition(relation(name, kind=kind)).propositions()
    return stated


def survey(**conditions: StatedTaskCondition) -> SetKindSurvey:
    """
    :param conditions: The condition each task of a single dataset states.
    :return: A survey over just those conditions, whose relations are RoboCasa's so that
        the shapes they describe are established.
    """
    return SetKindSurvey(
        datasets=[
            SurveyedDataset(
                name="dataset",
                conditions=dict(conditions),
                geometry=RoboCasaPredicateGeometry(),
            )
        ]
    )


# %% the shape a predicate describes


def test_marker_marks_the_text_it_occurs_in():
    """
    A suite writes its measure as part of a longer name, so a marker marks whatever text
    it occurs in rather than only the text it equals.
    """
    marker = GeometryMarker("norm", PredicateGeometry.EUCLIDEAN_DISC)

    assert marker.marks("np.linalg.norm")
    assert not marker.marks("np.linalg.det")


def test_predicate_over_a_finite_domain_is_represented_exactly():
    """
    A discrete predicate holds on a set of states rather than a region, so no shape is
    approximated and the representation captures all of it.
    """
    geometry = RoboCasaPredicateGeometry().geometry_of(proposition("OU.door_is_open"))

    assert geometry is PredicateGeometry.SYMBOLIC
    assert geometry.intersection_over_union == 1.0


def test_threshold_on_a_three_dimensional_distance_is_a_ball():
    """
    A predicate thresholding a distance in space describes a ball, which fills only part
    of the box representing it.
    """
    geometry = RoboCasaPredicateGeometry().geometry_of(
        proposition("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)
    )

    assert geometry is PredicateGeometry.EUCLIDEAN_BALL
    assert geometry.intersection_over_union == math.pi / 6.0


def test_threshold_on_a_horizontal_distance_is_a_disc():
    """
    A predicate thresholding a distance taken in the horizontal plane describes a disc,
    which fills more of its bounding box than a ball does of its own.
    """
    geometry = RoboCasaPredicateGeometry().geometry_of(
        proposition("np.linalg.norm", kind=PredicateKind.CONTINUOUS)
    )

    assert geometry is PredicateGeometry.EUCLIDEAN_DISC
    assert geometry.intersection_over_union == math.pi / 4.0


def test_bound_taken_per_axis_is_represented_exactly():
    """
    A predicate bounding a quantity per axis already describes a box, so representing it
    as one loses nothing.
    """
    geometry = RoboCasaPredicateGeometry().geometry_of(
        proposition("OU.obj_fixture_bbox_min_dist", kind=PredicateKind.CONTINUOUS)
    )

    assert geometry is PredicateGeometry.AXIS_ALIGNED
    assert geometry.intersection_over_union == 1.0


def test_unrecognised_predicate_is_assigned_no_shape():
    """
    A predicate whose shape the survey does not establish is left unclassified rather
    than assigned one it was not shown to have.
    """
    vocabulary = RoboCasaPredicateGeometry()

    assert (
        vocabulary.geometry_of(
            proposition("some.unknown.measure", kind=PredicateKind.CONTINUOUS)
        )
        is PredicateGeometry.UNCLASSIFIED
    )
    assert (
        vocabulary.geometry_of(
            proposition("some.unknown.check", kind=PredicateKind.UNCLASSIFIED)
        )
        is PredicateGeometry.UNCLASSIFIED
    )


def test_vocabulary_of_no_suite_establishes_no_shape():
    """
    A shape follows from the quantity a suite measures, so a vocabulary belonging to no
    suite establishes none -- rather than silently applying one suite's markers to
    another suite's predicates.
    """
    vocabulary = PredicateGeometryVocabulary()

    assert (
        vocabulary.geometry_of(
            proposition("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)
        )
        is PredicateGeometry.UNCLASSIFIED
    )


# %% how often each kind occurs


def test_each_kind_is_counted_and_reported_as_a_share_of_the_dataset():
    """
    The frequencies say what part of a dataset each kind of set makes up, which is what
    the survey is for.
    """
    frequencies = survey(
        first=condition(relation("OU.door_is_open")),
        second=condition(relation("OU.drawer_is_open")),
        third=condition(relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)),
        fourth=condition(relation("OU.stove_is_on")),
    ).run()

    by_kind = {frequency.set_kind: frequency for frequency in frequencies}
    assert by_kind[ConditionKind.DISCRETE].condition_count == 3
    assert by_kind[ConditionKind.DISCRETE].percentage == 75.0
    assert by_kind[ConditionKind.CONTINUOUS].condition_count == 1
    assert by_kind[ConditionKind.CONTINUOUS].percentage == 25.0


def test_kinds_are_reported_most_frequent_first():
    """
    Reporting the most frequent kind first states what a dataset mostly consists of
    without the reader having to compare the rows.
    """
    frequencies = survey(
        first=condition(relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)),
        second=condition(relation("OU.door_is_open")),
        third=condition(relation("OU.drawer_is_open")),
    ).run()

    assert [frequency.set_kind for frequency in frequencies] == [
        ConditionKind.DISCRETE,
        ConditionKind.CONTINUOUS,
    ]


def test_datasets_are_surveyed_separately():
    """
    Each dataset is counted on its own, so one dataset's conditions never enter
    another's shares.
    """
    frequencies = SetKindSurvey(
        datasets=[
            SurveyedDataset(
                name="first",
                conditions={"task": condition(relation("OU.door_is_open"))},
            ),
            SurveyedDataset(
                name="second",
                conditions={
                    "task": condition(
                        relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)
                    )
                },
            ),
        ]
    ).run()

    assert [(each.dataset, each.set_kind, each.percentage) for each in frequencies] == [
        ("first", ConditionKind.DISCRETE, 100.0),
        ("second", ConditionKind.CONTINUOUS, 100.0),
    ]


# %% what representing a kind costs and captures


def test_average_simple_sets_counts_the_terms_a_representation_needs():
    """
    Requiring several predicates at once stays one term while accepting either needs
    two, so the average says how many terms a condition of a kind typically costs.
    """
    frequencies = survey(
        conjunctive=condition(
            and_(relation("OU.door_is_open"), relation("OU.drawer_is_open"))
        ),
        disjunctive=condition(
            or_(relation("OU.stove_is_on"), relation("OU.sink_is_on"))
        ),
    ).run()

    assert frequencies[0].average_simple_sets == 1.5


def test_average_fidelity_is_taken_over_the_predicates_applied():
    """
    A condition's fidelity follows from the shapes its predicates describe, so a
    condition requiring an exact state and an approximated distance sits between the
    two.
    """
    frequencies = survey(
        task=condition(
            and_(
                relation("OU.door_is_open"),
                relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS),
            )
        )
    ).run()

    assert frequencies[0].average_intersection_over_union == (1.0 + math.pi / 6.0) / 2.0


def test_predicates_of_unestablished_shape_are_left_out_of_the_average():
    """
    A predicate the survey assigns no shape carries no fidelity, so it must not enter
    the average as though its representation captured nothing.
    """
    frequencies = survey(
        task=condition(
            and_(
                relation("OU.door_is_open"),
                relation("some.unknown.measure", kind=PredicateKind.CONTINUOUS),
            )
        )
    ).run()

    assert frequencies[0].average_intersection_over_union == 1.0


def test_fidelity_of_conditions_without_an_established_shape_is_reported_as_absent():
    """
    A kind whose predicates all have unestablished shapes has no fidelity to report, so
    it reports none rather than a zero that would read as capturing nothing.
    """
    frequencies = survey(
        task=condition(relation("some.unknown.check", kind=PredicateKind.UNCLASSIFIED))
    ).run()

    assert frequencies[0].average_intersection_over_union is None


def test_first_order_conditions_are_counted_and_left_out_of_the_representation_cost():
    """
    A condition quantifying over a collection is first order, which a propositional
    algebra represents no statement of, so it is reported as such and the cost of
    representing the kind is taken over the conditions that are propositional.
    """
    quantified = condition(
        exists(SceneObject.provided_by_the_scene(), relation("OU.obj_in_receptacle"))
    )
    propositional = condition(
        or_(relation("OU.door_is_open"), relation("OU.drawer_is_open"))
    )

    frequencies = survey(first=quantified, second=propositional).run()

    assert frequencies[0].condition_count == 2
    assert frequencies[0].first_order_condition_count == 1
    assert frequencies[0].average_simple_sets == 2.0


def test_first_order_conditions_still_count_towards_their_kind():
    """
    Which kind of set a condition describes follows from the predicates it applies and
    not from whether it quantifies, so a first order condition is counted in its kind's
    share of the dataset like any other.
    """
    quantified = condition(
        for_all(
            SceneObject.provided_by_the_scene(),
            relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS),
        )
    )

    frequencies = survey(
        first=quantified, second=condition(relation("OU.door_is_open"))
    ).run()

    by_kind = {frequency.set_kind: frequency for frequency in frequencies}
    assert by_kind[ConditionKind.CONTINUOUS].condition_count == 1
    assert by_kind[ConditionKind.CONTINUOUS].percentage == 50.0
    assert by_kind[ConditionKind.DISCRETE].first_order_condition_count == 0


def test_kind_stated_only_by_first_order_conditions_reports_no_cost():
    """
    A kind whose conditions all quantify has no propositional representation to measure,
    so it reports none rather than a cost taken over an invented number of objects.
    """
    quantified = condition(
        exists(SceneObject.provided_by_the_scene(), relation("OU.obj_in_receptacle"))
    )

    frequencies = survey(task=quantified).run()

    assert frequencies[0].first_order_condition_count == 1
    assert frequencies[0].average_simple_sets is None


def test_average_that_comes_out_whole_is_still_reported_as_a_quantity():
    """
    An average is a quantity whether or not it divides exactly, so a column of averages
    reads uniformly instead of mixing counts with quantities.
    """
    frequencies = survey(
        first=condition(relation("OU.door_is_open")),
        second=condition(relation("OU.drawer_is_open")),
    ).run()

    assert isinstance(frequencies[0].average_simple_sets, float)
