"""
This module provides functionality to build an HSRB robot world, create robot
action plans, execute plans in a simulated environment, and measure the
performance of data operations, such as serialization and database storage.

It includes tools for running experiments to evaluate the reliability
and performance of robotic action plans and data operations.
"""

import pathlib
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import plotly.graph_objects as go
import tqdm
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

import coraplex.orm.ormatic_interface  # type: ignore  # noqa: F401
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.orm.ormatic_interface import Base, PlanMappingDAO  # type: ignore
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.ormatic.data_access_objects.dao import selectin_loading
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

_REPO_ROOT = pathlib.Path(__file__).parents[4]
_APARTMENT_URDF = _REPO_ROOT / "coraplex" / "resources" / "worlds" / "apartment.urdf"


def _build_hsrb_world() -> World:
    urdf_parser = URDFParser.from_file(file_path=HSRB.get_ros_file_path())
    world = urdf_parser.parse()
    HSRB.from_world(world)

    with world.modify_world():
        old_root = world.root
        map_body = Body(name=PrefixedName("map"))
        odom = Body(name=PrefixedName("odom_combined"))
        map_C_odom = Connection6DoF.create_with_dofs(world, map_body, odom)
        world.add_connection(map_C_odom)
        drive = OmniDrive.create_with_dofs(parent=odom, child=old_root, world=world)
        world.add_connection(drive)
        drive.has_hardware_interface = True

    return world


def build_cram_world():
    """
    Build an HSRB+apartment world and return ``(world, hsrb, context)``.
    """
    hsrb_world = _build_hsrb_world()
    apartment = URDFParser.from_file(str(_APARTMENT_URDF)).parse()
    hsrb_world.merge_world(apartment)
    hsrb_world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    hsrb = hsrb_world.get_semantic_annotations_by_type(HSRB)[0]
    ctx = Context(
        hsrb_world,
        hsrb,
        evaluate_conditions=False,
        query_backend=ProbabilisticBackend(),
    )
    return hsrb_world, hsrb, ctx


def _random_navigate_action(world: World):
    """
    Return an underspecified :class:`NavigateAction` with randomised pose.
    """
    action = underspecified(NavigateAction)(
        target_location=underspecified(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=...,
            reference_frame=world.root,
        ),
        keep_joint_states=True,
    )
    action.expression.limit(10)
    return action


def create_plan(world: World, ctx: Context, n_actions: int):
    """
    Create a sequential plan with *n_actions* random :class:`NavigateAction`
    instances.
    """
    actions = [_random_navigate_action(world) for _ in range(n_actions)]
    return sequential(actions, context=ctx).plan


@dataclass
class ORMaticReliabilityExperimentResult(ExperimentResult):
    """
    Raw timings from a single plan-creation, execution and DB-write cycle.
    """

    plan_size: int
    """
    Number of actions in the plan.
    """

    world_building_duration: float
    """
    Seconds to build the PR2+apartment world.
    """

    plan_execution_duration: float
    """
    Seconds to execute the plan under simulated_robot.
    """

    to_data_access_object_duration: float
    """
    Seconds for to_dao() to serialise the plan to a DAO.
    """

    writing_to_database_duration: float
    """
    Combined seconds for session.add(dao) + session.commit().
    """

    reading_from_database_duration: float
    """
    Seconds for session.scalars(select(PlanMappingDAO)).one().
    """

    reconstruction_duration: float
    """
    Seconds for fetched_plan.from_dao().
    """


@dataclass
class ORMaticReliabilityAggregateResult(ExperimentResult):
    """
    Aggregated timings across multiple reliability experiment runs.
    """

    plan_size: int
    """
    Number of actions in each plan in this aggregate.
    """

    world_building_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of world-building time across runs (seconds).
    """

    plan_execution_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of plan execution time under simulated_robot
    (seconds).
    """

    to_data_access_object_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of to_dao() serialisation time (seconds).
    """

    writing_to_database_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of session.add() + session.commit() time
    (seconds).
    """

    reading_from_database_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of
    session.scalars(select(PlanMappingDAO)).one() time (seconds).
    """

    reconstruction_duration: MeanAndStandardDeviation
    """
    Mean and standard deviation of from_dao() reconstruction time (seconds).
    """


def reliability_experiment(
    plan_size: int,
    world: World,
    context: Context,
    world_building_duration: float,
) -> ORMaticReliabilityExperimentResult:
    """
    Run a single reliability iteration: create and execute a plan, then write
    it to an in-memory SQLite database, measuring each phase separately.

    :param plan_size: Number of actions to include in the random plan.
    :param world: The pre-built world to create the plan in.
    :param context: The execution context.
    :param world_building_duration: Pre-measured world building time
        (s).
    :return: Timing breakdown for this single run.
    """
    plan = create_plan(world, context, plan_size)

    t0 = time.perf_counter()
    with simulated_robot:
        plan.perform()
    plan_execution_duration = time.perf_counter() - t0

    t0 = time.perf_counter()
    dao = to_dao(plan)
    to_dao_duration = time.perf_counter() - t0

    engine = create_engine("sqlite:///:memory:")
    session = sessionmaker(engine)()
    Base.metadata.create_all(bind=session.bind)

    t0 = time.perf_counter()
    session.add(dao)
    session.commit()
    writing_to_database_duration = time.perf_counter() - t0

    with selectin_loading(session):
        t0 = time.perf_counter()
        fetched = session.scalars(select(PlanMappingDAO)).one()
        reading_from_database_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        fetched.from_dao()
        reconstruction_duration = time.perf_counter() - t0

    drop_database(session.bind)
    session.close()
    engine.dispose()

    return ORMaticReliabilityExperimentResult(
        plan_size=plan_size,
        world_building_duration=round(world_building_duration, 3),
        plan_execution_duration=round(plan_execution_duration, 3),
        to_data_access_object_duration=round(to_dao_duration, 3),
        writing_to_database_duration=round(writing_to_database_duration, 3),
        reading_from_database_duration=round(reading_from_database_duration, 3),
        reconstruction_duration=round(reconstruction_duration, 3),
    )


def run_reliability_experiment(
    plan_size: int,
    iterations: int = 10,
) -> tuple[ORMaticReliabilityAggregateResult, List[ORMaticReliabilityExperimentResult]]:
    """
    Run *iterations* reliability experiment cycles for plans of *plan_size*
    actions and aggregate the results.

    The world is rebuilt from scratch on every iteration so that
    ``world_building_duration`` is measured repeatedly and yields a meaningful
    mean and standard deviation.

    :param plan_size: Number of actions per plan.
    :param iterations: Number of independent iterations to aggregate.
    :return: Tuple of (aggregate result, raw per-iteration results).
    """
    raw = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        world, _, ctx = build_cram_world()
        world_building_duration = time.perf_counter() - t0

        result = reliability_experiment(plan_size, world, ctx, world_building_duration)
        raw.append(result)

    aggregate = ORMaticReliabilityAggregateResult(
        plan_size=plan_size,
        world_building_duration=MeanAndStandardDeviation.from_measurements(
            [r.world_building_duration for r in raw]
        ),
        plan_execution_duration=MeanAndStandardDeviation.from_measurements(
            [r.plan_execution_duration for r in raw]
        ),
        to_data_access_object_duration=MeanAndStandardDeviation.from_measurements(
            [r.to_data_access_object_duration for r in raw]
        ),
        writing_to_database_duration=MeanAndStandardDeviation.from_measurements(
            [r.writing_to_database_duration for r in raw]
        ),
        reading_from_database_duration=MeanAndStandardDeviation.from_measurements(
            [r.reading_from_database_duration for r in raw]
        ),
        reconstruction_duration=MeanAndStandardDeviation.from_measurements(
            [r.reconstruction_duration for r in raw]
        ),
    )
    return aggregate, raw


def plot_reliability(
    raw_results: List[ORMaticReliabilityExperimentResult],
) -> go.Figure:
    """
    Produce a grouped violin plot of plan size vs duration for each timing
    phase.

    :param raw_results: All per-iteration
        :class:`ORMaticReliabilityExperimentResult` instances across
        every plan size.
    :return: A Plotly figure ready for display or export.
    """
    phases = [
        ("plan_execution_duration", "Plan Execution"),
        ("to_data_access_object_duration", "to_dao()"),
        ("writing_to_database_duration", "Write to DB"),
        ("reading_from_database_duration", "Read from DB"),
        ("reconstruction_duration", "from_dao()"),
    ]

    traces = []
    for field, label in phases:
        traces.append(
            go.Violin(
                x=[str(r.plan_size) for r in raw_results],
                y=[getattr(r, field) for r in raw_results],
                name=label,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        violinmode="group",
        xaxis_title="Plan Size (actions)",
        yaxis_title="Duration (s)",
        title="ORMatic Reliability: Plan Size vs Timing",
    )
    return fig


def main():
    aggregate_results = []
    all_raw: List[ORMaticReliabilityExperimentResult] = []
    for plan_size in tqdm.tqdm(
        [
            1,
            3,
        ]
    ):
        aggregate, raw = run_reliability_experiment(plan_size, iterations=10)
        aggregate_results.append(aggregate)
        all_raw.extend(raw)

    table = ExperimentsTable(aggregate_results)
    print(TypstRenderer(table).render_table())
    plot_reliability(all_raw).show()


if __name__ == "__main__":
    main()
