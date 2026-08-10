"""
This module surveys which kinds of set occur in a dataset's task conditions, how often,
and what representing them as a random event costs and captures.

One row per dataset and kind of set, giving the number of conditions of that kind, the
percentage of the dataset's conditions they are, how many of them are first order, the
average number of simple sets their representation needs, and the average intersection
over union between that representation and the region a condition's predicates really
describe.

A condition quantifying over the objects a scene provides is first order, and the
product algebra is propositional, so such a condition is counted and reported but never
translated: what it costs to represent would otherwise be decided by how many objects
the survey chose to instantiate it over.

The intersection over union follows from the shape a predicate describes. A predicate
over a finite set of states, and one bounding a quantity per axis, are both represented
exactly. A predicate thresholding a Euclidean distance is a ball, which no *single* axis-aligned
box captures exactly: in three dimensions a ball fills ``pi / 6`` of the box bounding it,
and in the horizontal plane a disc fills ``pi / 4`` of its bounding square. Spending more
than one box captures more of it, and
:mod:`experiments.random_events_experiments.curved_region_approximation` measures how much
more, so the fidelity reported here is what one box per predicate buys and not a limit.
"""

from __future__ import annotations

import collections
import enum
import math
import pathlib
import statistics
from dataclasses import dataclass, field

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from experiments.random_events_experiments.task_condition_survey.condition_event_translation import (
    ConditionEventTranslator,
)
from experiments.random_events_experiments.task_condition_survey.condition_extraction import (
    TaskConditionReader,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConditionKind,
    PredicateKind,
    StatedProposition,
    StatedTaskCondition,
)
from experiments.random_events_experiments.task_condition_survey.mani_skill_condition_survey import (
    ManiSkillConditionSurvey,
)
from experiments.random_events_experiments.task_condition_survey.rlbench_condition_survey import (
    RLBenchConditionSurvey,
)
from experiments.random_events_experiments.task_condition_survey.robocasa_condition_survey import (
    RoboCasaConditionSurvey,
)

# %% what shape a predicate describes


class PredicateGeometry(enum.Enum):
    """
    The shape of the region a predicate describes, which decides how much of it an axis-
    aligned representation captures.
    """

    SYMBOLIC = "a finite set of states"
    """
    Represented exactly, since a symbolic set is a set of values and not a region.
    """

    AXIS_ALIGNED = "a bound on a quantity per axis"
    """
    Represented exactly, since the region is already a box.
    """

    EUCLIDEAN_BALL = "a threshold on a distance in three dimensions"
    """
    A ball, which fills ``pi / 6`` of the box bounding it.
    """

    EUCLIDEAN_DISC = "a threshold on a distance in the horizontal plane"
    """
    A disc, which fills ``pi / 4`` of the square bounding it.
    """

    UNCLASSIFIED = "a region whose shape this survey does not establish"
    """
    Left out of the average rather than assigned a shape it was not shown to have.
    """

    @property
    def intersection_over_union(self) -> float | None:
        """
        :return: The fraction of the region's axis-aligned representation that the region
            itself fills, or ``None`` when the shape was not established.
        """
        return {
            PredicateGeometry.SYMBOLIC: 1.0,
            PredicateGeometry.AXIS_ALIGNED: 1.0,
            PredicateGeometry.EUCLIDEAN_BALL: math.pi / 6.0,
            PredicateGeometry.EUCLIDEAN_DISC: math.pi / 4.0,
            PredicateGeometry.UNCLASSIFIED: None,
        }[self]


@dataclass(frozen=True)
class GeometryMarker:
    """
    Source text a suite writes, and the shape a predicate written with it describes.
    """

    source_text: str
    """
    The text this marker is recognised by.
    """

    geometry: PredicateGeometry
    """
    The shape a predicate written with it describes.
    """

    def marks(self, written_as: str) -> bool:
        """
        :param written_as: The part of a proposition this marker is about, as the task
            writes it.
        :return: Whether the marker occurs in it.
        """
        return self.source_text in written_as


@dataclass
class PredicateGeometryVocabulary:
    """
    Establishes the shape each predicate describes, from what it is applied under.

    A suite takes its measures over its own quantities, so a suite supplies its own
    markers. Establishing no shape is the honest default: a predicate whose shape is not
    stated is left out of the fidelity average rather than assigned one it was not shown
    to have.
    """

    shapes_by_application: tuple[GeometryMarker, ...] = ()
    """
    Shape assigned by what a measure was applied to, consulted before the measure's own
    name because slicing a position changes the dimension of the region it describes.
    """

    shapes_by_measure: tuple[GeometryMarker, ...] = ()
    """
    Shape assigned by the name a measure is taken under, consulted in order.
    """

    def geometry_of(self, proposition: StatedProposition) -> PredicateGeometry:
        """
        What the measure was applied to decides first, since the same measure taken over a
        whole position and over a slice of one describes regions of different dimension.
        Each set of markers is matched only against the part of the proposition it is about,
        so a marker naming a measure cannot be matched by an object's name.

        :param proposition: A proposition a condition states.
        :return: The shape of the region it describes.
        """
        if proposition.kind is PredicateKind.DISCRETE:
            return PredicateGeometry.SYMBOLIC
        if proposition.kind is PredicateKind.UNCLASSIFIED:
            return PredicateGeometry.UNCLASSIFIED
        for marker in self.shapes_by_application:
            if marker.marks(proposition.stated_of):
                return marker.geometry
        for marker in self.shapes_by_measure:
            if marker.marks(proposition.stated_under):
                return marker.geometry
        return PredicateGeometry.UNCLASSIFIED


@dataclass
class RoboCasaPredicateGeometry(PredicateGeometryVocabulary):
    """
    Establishes the shape each predicate RoboCasa applies describes.

    Every distance RoboCasa thresholds inside a condition is taken in the horizontal
    plane except the two gripper clearances, which are taken in space.
    """

    shapes_by_measure: tuple[GeometryMarker, ...] = (
        GeometryMarker("gripper_obj_far", PredicateGeometry.EUCLIDEAN_BALL),
        GeometryMarker("gripper_fxtr_far", PredicateGeometry.EUCLIDEAN_BALL),
        GeometryMarker("norm", PredicateGeometry.EUCLIDEAN_DISC),
        GeometryMarker("bbox_min_dist", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("abs", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("min", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("max", PredicateGeometry.AXIS_ALIGNED),
    )
    """
    Shape assigned by the name a measure is taken under, consulted in order.
    """


@dataclass
class ManiSkillPredicateGeometry(PredicateGeometryVocabulary):
    """
    Establishes the shape each predicate ManiSkill applies describes.

    ManiSkill takes a distance both over a whole position and over a position sliced to
    its first two axes, so the shape follows from what the measure was applied to rather
    than from the measure's name alone.
    """

    shapes_by_application: tuple[GeometryMarker, ...] = (
        GeometryMarker(":2]", PredicateGeometry.EUCLIDEAN_DISC),
        GeometryMarker(":1]", PredicateGeometry.AXIS_ALIGNED),
    )
    """
    A distance taken over a position sliced to two axes is a distance in a plane, and
    one sliced to a single axis is a bound on that axis.
    """

    shapes_by_measure: tuple[GeometryMarker, ...] = (
        GeometryMarker("is_static", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("norm", PredicateGeometry.EUCLIDEAN_BALL),
        GeometryMarker("abs", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("max", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("min", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("qpos", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("qvel", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("pose.p", PredicateGeometry.AXIS_ALIGNED),
    )
    """
    Shape assigned by the name a measure is taken under, consulted in order.
    """


@dataclass
class RLBenchPredicateGeometry(PredicateGeometryVocabulary):
    """
    Establishes the shape each condition RLBench registers describes.

    Its continuous conditions bound one coordinate each: a joint's displacement is a
    bound on a single axis, while a deviation from a path is a distance in space.
    """

    shapes_by_measure: tuple[GeometryMarker, ...] = (
        GeometryMarker("JointCondition", PredicateGeometry.AXIS_ALIGNED),
        GeometryMarker("FollowCondition", PredicateGeometry.EUCLIDEAN_BALL),
    )
    """
    Shape assigned to each of the suite's continuous conditions.
    """


# %% set kind frequencies


@dataclass
class SetKindFrequency(ExperimentResult):
    """
    How often one kind of set occurs in one dataset, and what representing it costs.
    """

    dataset: str
    """
    Dataset the conditions were read from.
    """

    set_kind: ConditionKind
    """
    Kind of set the conditions describe.
    """
    condition_count: int
    """
    Number of conditions describing a set of that kind.
    """
    percentage: float
    """
    Percentage of the dataset's conditions that describe a set of that kind.
    """

    first_order_condition_count: int
    """
    Number of those conditions that quantify over the objects a scene provides, and so
    state something no propositional representation captures.
    """
    average_simple_sets: float | None
    """
    Average number of simple sets the random event representing such a condition needs,
    taken over the conditions that are propositional, or ``None`` when all of them
    quantify.
    """

    average_intersection_over_union: float | None
    """
    Average, over the predicates these conditions apply, of the fraction of a
    predicate's axis-aligned representation that the region it describes fills, or
    ``None`` when none of those predicates has an established shape.
    """


@dataclass
class SurveyedDataset:
    """
    One dataset's conditions, together with what its own predicates describe.

    Two datasets name their predicates differently and take their measures over
    different quantities, so the shape a predicate describes is established per dataset
    rather than once for all of them.
    """

    name: str
    """
    Name the dataset is reported under.
    """

    conditions: dict[str, StatedTaskCondition]
    """
    The condition each of its tasks states, keyed by the name of the class defining the
    task.

    A key is the name read out of the suite's source rather than a member of an
    enumeration of tasks, since which tasks a suite defines is settled by the version of
    it that happens to be installed.
    """

    geometry: PredicateGeometryVocabulary = field(
        default_factory=PredicateGeometryVocabulary
    )
    """
    Establishes the shape each of its predicates describes.
    """


@dataclass
class SetKindSurvey:
    """
    Counts the kinds of set occurring in each of a number of datasets, and measures what
    representing them costs and captures.
    """

    datasets: list[SurveyedDataset]
    """
    The datasets to survey, reported in the order given.
    """

    def run(self) -> list[SetKindFrequency]:
        """
        :return: One entry per dataset and kind of set, most frequent first within each
            dataset.
        """
        return [
            frequency
            for dataset in self.datasets
            for frequency in self.frequencies_of(dataset)
        ]

    def frequencies_of(self, dataset: SurveyedDataset) -> list[SetKindFrequency]:
        """
        :param dataset: A dataset's conditions and the shapes its predicates describe.
        :return: One entry per kind of set occurring in that dataset.
        """
        by_kind: dict[ConditionKind, list[StatedTaskCondition]] = (
            collections.defaultdict(list)
        )
        for condition in dataset.conditions.values():
            by_kind[condition.kind].append(condition)

        total = sum(len(group) for group in by_kind.values())
        return [
            SetKindFrequency(
                dataset=dataset.name,
                set_kind=set_kind,
                condition_count=len(group),
                percentage=100.0 * len(group) / total,
                first_order_condition_count=len(self.first_order(group)),
                average_simple_sets=self.average_simple_sets(group),
                average_intersection_over_union=self.average_fidelity(
                    group, dataset.geometry
                ),
            )
            for set_kind, group in sorted(
                by_kind.items(), key=lambda entry: -len(entry[1])
            )
        ]

    @staticmethod
    def first_order(
        conditions: list[StatedTaskCondition],
    ) -> list[StatedTaskCondition]:
        """
        :param conditions: Conditions of one kind.
        :return: Those of them that quantify over the objects a scene provides.
        """
        return [condition for condition in conditions if condition.quantifications()]

    def average_simple_sets(
        self, conditions: list[StatedTaskCondition]
    ) -> float | None:
        """
        A quantified condition is left out rather than instantiated over a chosen number
        of objects, since the number chosen would then decide the cost reported.

        :param conditions: Conditions of one kind.
        :return: The average number of simple sets their representations need, or
            ``None`` when every one of them quantifies.
        """
        translator = ConditionEventTranslator()
        propositional = [
            condition for condition in conditions if not condition.quantifications()
        ]
        if not propositional:
            return None
        return float(
            statistics.mean(
                translator.simple_set_count(condition) for condition in propositional
            )
        )

    def average_fidelity(
        self,
        conditions: list[StatedTaskCondition],
        geometry: PredicateGeometryVocabulary,
    ) -> float | None:
        """
        :param conditions: Conditions of one kind.
        :param geometry: Establishes the shape each of the dataset's predicates
            describes.
        :return: The average, over the propositions they state, of how much of a
            proposition's axis-aligned representation the region it describes fills, or
            ``None`` when none of them has an established shape.
        """
        fidelities = [
            geometry.geometry_of(proposition).intersection_over_union
            for condition in conditions
            for proposition in condition.propositions()
        ]
        established = [each for each in fidelities if each is not None]
        return statistics.mean(established) if established else None


# %% experiment entry point


def stated_conditions(reader: TaskConditionReader) -> dict[str, StatedTaskCondition]:
    """
    :param reader: Reads one suite's conditions.
    :return: The conditions of that suite that were read in full and state at least one
        proposition, which are the ones a kind can be established for.
    """
    return {
        task: condition
        for task, condition in reader.conditions().items()
        if condition.is_fully_read() and condition.propositions()
    }


def main() -> None:
    """
    Surveys every installed suite and prints the table, recording the rows alongside it.
    """
    output_directory = pathlib.Path("set_kind_survey_output")
    output_directory.mkdir(parents=True, exist_ok=True)

    datasets = [
        SurveyedDataset(
            name="RoboCasa",
            conditions=stated_conditions(
                RoboCasaConditionSurvey.for_installed_robocasa()
            ),
            geometry=RoboCasaPredicateGeometry(),
        ),
        SurveyedDataset(
            name="ManiSkill",
            conditions=stated_conditions(
                ManiSkillConditionSurvey.for_installed_mani_skill()
            ),
            geometry=ManiSkillPredicateGeometry(),
        ),
        SurveyedDataset(
            name="RLBench",
            conditions=stated_conditions(
                RLBenchConditionSurvey.for_installed_rlbench()
            ),
            geometry=RLBenchPredicateGeometry(),
        ),
    ]

    table = ExperimentsTable(SetKindSurvey(datasets=datasets).run())
    table.write_manifest(output_directory, "set_kind_frequencies.json")
    print(
        TypstRenderer(table).render_figure(
            "Kinds of set occurring in the task conditions of each dataset, the "
            "percentage of that dataset's conditions describing each kind, how many of "
            "them quantify over the objects a scene provides and so are first order, "
            "the average number of simple sets the random event representing the "
            "remaining ones needs, and the average fraction of that representation the "
            "described region fills."
        )
    )


if __name__ == "__main__":
    main()
