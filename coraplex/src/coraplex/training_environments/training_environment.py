import json
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import Any, Generic, TypeVar

from krrood.adapters.json_serializer import from_json
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, an, variable
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import (
    fully_factorized,
    multiply_distributions,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProductUnit,
)
from coraplex.plans.failures import (
    PlanFailure,
    EmptyUnderspecified,
)
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.core.misc import MoveToReach
from cramph.composites import StatementNode
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import (
    EndEffector,
    AbstractRobot,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import (
    Point2,
    Point3,
    Pose,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

TrainedAction = TypeVar("TrainedAction", bound=Action)


@dataclass
class TrainingEpisode:
    """
    The underspecified action one episode of training executes, and the executor
    executing it.
    """

    action: StatementNode
    """
    The underspecified action whose grounded variants the episode tries.
    """

    executor: KinematicallySimulatedPlanExecutor
    """
    The executor running the action in the world built for the episode.
    """


@dataclass
class TrainingEnvironment(Generic[TrainedAction], SubClassSafeGeneric, ABC):
    """
    A training environment for generating data for the parameterization of the actions
    of the type it binds.
    """

    executed_plans: list[StatechartNode] = field(default_factory=list)
    """
    The executed plans during training.
    """

    tried_actions: list[Action] = field(default_factory=list)
    """
    Every action the executed plans tried, whether it succeeded or not.
    """

    visualize: bool = False
    """
    Rather create a visualization of the executed plans or not.
    """

    @property
    def action_type(self) -> type[TrainedAction]:
        """
        :return: The type of action that is trained.
        """
        return type(self).get_type_of_generic_parameter(TrainedAction)

    @abstractmethod
    def setup_world(self, *kwargs) -> World:
        """
        :return: A world containing everything thats needed for training, including the robot.
        """

    @abstractmethod
    def setup_episode(self, limit: int = 10, **kwargs: Any) -> TrainingEpisode:
        """
        Create an underspecified action, in a world of its own, whose grounded variants
        the episode tries.

        :param limit: The maximum number of actions that should be executed.
        :return: The action and the executor executing it.
        """

    def generate_episodes(self, number_of_actions: int = 10):
        """
        Generate episodes until `number_of_actions` have been executed.

        :param number_of_actions: The number of action executions that should be
            generated.
        """
        remaining_actions = number_of_actions

        while remaining_actions > 0:
            executed_actions = self.generate_episode(remaining_actions)
            remaining_actions -= executed_actions

    def generate_episode(self, limit: int) -> int:
        """
        Generate a single episode.

        :param limit: The maximum number of actions that should be executed.
        :return: The number of actions executed in the episode.
        """
        episode = self.setup_episode(limit)

        if self.visualize:
            import rclpy

            pub = VizMarkerPublisher(
                _world=episode.executor.context.world,
                node=rclpy.create_node("test_node"),
            )

        executor = episode.executor
        statechart = Statechart(context=executor.context)
        statechart.add_node(episode.action)
        try:
            executor.compile(statechart)
            executor.execute()
        except EmptyUnderspecified:
            # No working solution found in this episode
            pass
        self.executed_plans.append(episode.action)

        self.tried_actions.extend(episode.action.grounded_nodes)
        number_of_executed_variants = len(episode.action.grounded_nodes)

        if self.visualize:
            pub.stop()
            pub.remove_from_world()

        return number_of_executed_variants


@dataclass
class MoveToReachTrainingEnvironment(TrainingEnvironment[MoveToReach]):
    """
    Training environment for MoveToReach actions in an empty world using the PR2.
    """

    model_path: Path | None = None
    """
    Path to a model file that should be used for the action.

    Creates a default model when this is None.
    """

    def setup_world(self, **kwargs) -> World:
        pr2_file = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

        urdf_parser = URDFParser.from_file(file_path=pr2_file)
        world_with_urdf = urdf_parser.parse()
        pr2 = PR2.from_world(world_with_urdf)

        with world_with_urdf.modify_world():
            old_root = world_with_urdf.root
            map = Body(name=PrefixedName("map"))
            localization_body = Body(name=PrefixedName("odom_combined"))

            map_C_localization = Connection6DoF.create_with_dofs(
                world_with_urdf, map, localization_body
            )
            world_with_urdf.add_connection(map_C_localization)

            c_root_bf = OmniDrive.create_with_dofs(
                parent=localization_body,
                child=old_root,
                world=world_with_urdf,
            )
            world_with_urdf.add_connection(c_root_bf)
            c_root_bf.has_hardware_interface = True

        return world_with_urdf

    def setup_episode(self, limit: int = 10, **kwargs: Any) -> TrainingEpisode:

        world = self.setup_world()
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        target_pose = Pose.from_xyz_rpy(
            x=0,
            y=0,
            z=1,
            reference_frame=world.root,
        )

        move_to_reach = a(MoveToReach)(
            reference_T_grasp=target_pose,
            target_pose_offset_robot=a(Pose2D)(
                position=a(Point2)(x=..., y=..., reference_frame=None),
                yaw=...,
                reference_frame=None,
            ),
            hip_rotation=...,
            end_effector=variable(EndEffector, world.semantic_annotations),
            approach_clearance=...,
        )

        move_to_reach.limit(limit)

        if self.model_path:
            query_backend = self.setup_backend_from_path(move_to_reach)
        else:
            query_backend = self.setup_backend(move_to_reach)

        executor = KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(
                world,
                [
                    RobotAccess(robot),
                    StatementGrounding(query_backend=query_backend),
                ],
            ),
        )

        return TrainingEpisode(
            action=StatementNode(statement=move_to_reach), executor=executor
        )

    def setup_backend(self, underspecified_action: Match) -> ProbabilisticBackend:
        """
        Create a backend containing the best guesses as distribution for this action in
        this environment.

        :param underspecified_action: The underspecified action to create a backend for.
        :return: The probabilistic backend
        """
        parameters = UnderspecifiedParameters(underspecified_action)
        robot_x = parameters.variable_of(
            underspecified_action.target_pose_offset_robot.position.x
        )
        robot_y = parameters.variable_of(
            underspecified_action.target_pose_offset_robot.position.y
        )
        hip_rotation = parameters.variable_of(underspecified_action.hip_rotation)
        approach_clearance = parameters.variable_of(
            underspecified_action.approach_clearance
        )

        distribution = fully_factorized(
            means={approach_clearance: 0.05},
            variances={robot_x: 0.5, robot_y: 0.5, hip_rotation: 0.1},
            variables=parameters.variables.values(),
        )

        hip_rotation_condition = SimpleEvent.from_data(
            {approach_clearance: closed(0.0, 0.4)}
        )
        hip_rotation_condition.fill_missing_variables(distribution.variables)
        distribution.log_truncated_of_simple_event_in_place(hip_rotation_condition)

        return ProbabilisticBackend(DictRegistry({self.action_type: distribution}))

    def setup_backend_from_path(self, underspecified_action: Match):
        """
        Setup a backend from a model file.

        Adds the variables of the action to the loaded model if they don't exist.

        :param underspecified_action: The action to load the model for.
        :return: The probabilistic backend.
        """
        with open(self.model_path) as f:
            distribution: ProbabilisticCircuit = from_json(json.load(f))

        # expand model with new variables
        parameters = UnderspecifiedParameters(underspecified_action)
        variables_not_in_parameters = set(parameters.variables.values()) - set(
            distribution.variables
        )
        distribution_for_variables_not_in_parameters = fully_factorized(
            variables_not_in_parameters
        )
        multiply_distributions(
            distribution, distribution_for_variables_not_in_parameters
        )

        return ProbabilisticBackend(DictRegistry({self.action_type: distribution}))
