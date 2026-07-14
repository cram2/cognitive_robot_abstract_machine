"""
Load a RoboCasa kitchen scene into a CRAM
:class:`~semantic_digital_twin.world.World`.

RoboCasa ships kitchens as robosuite/MuJoCo assets. This demo drives the
:class:`~semantic_digital_twin.adapters.robocasa_dataset.loader.RoboCasaDatasetLoader`
adapter to compose one such kitchen, parse it into CRAM's semantic world model, and
report the bodies and semantic annotations the adapter attached (cabinets, drawers,
handles, doors, ...). With ``--robot`` it also spawns a PR2 at a counter and has it pick
up an apple resting on the counter surface, and with ``--visualize`` it publishes the
world as RViz markers so the kitchen can be inspected visually.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.robocasa_kitchen_demo
    python -m experiments.robocasa_kitchen_demo --layout LAYOUT005 --style STYLE003
    python -m experiments.robocasa_kitchen_demo --robot --visualize

.. note::
    The kitchen assets must be downloaded once via
    ``python -m robocasa.scripts.download_kitchen_assets`` before this demo can build a
    scene.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.failures import PlanFailure
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaObjectCategory,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    CounterTop,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    rclpy = None

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedBody:
    """
    One semantic annotation the loader attached to the kitchen, described for display.
    """

    annotation_type: str
    """
    Name of the :class:`SemanticAnnotation` subclass (for example ``"HingeCabinet"``).
    """

    body_name: str
    """
    Name of the world body the annotation is rooted at.
    """

    def __str__(self) -> str:
        return f"{self.annotation_type:<28} -> {self.body_name}"


@dataclass
class KitchenSceneSummary:
    """
    A human-readable summary of a loaded kitchen world.
    """

    layout: LayoutType
    """
    The RoboCasa layout the kitchen was composed from.
    """

    style: StyleType
    """
    The RoboCasa visual style the kitchen was composed from.
    """

    number_of_bodies: int
    """
    Total number of bodies in the parsed world.
    """

    annotated_bodies: List[AnnotatedBody]
    """
    The semantically annotated bodies, one entry per attached annotation.
    """

    @classmethod
    def from_world(
        cls, world: World, layout: LayoutType, style: StyleType
    ) -> KitchenSceneSummary:
        """
        Build a summary from a loaded kitchen world.

        :param world: The world produced by the RoboCasa loader.
        :param layout: The layout the kitchen was composed from.
        :param style: The style the kitchen was composed from.
        :return: The summary.
        """
        annotated_bodies = [
            AnnotatedBody(
                annotation_type=type(annotation).__name__,
                body_name=annotation.root.name.name,
            )
            for annotation in world.semantic_annotations
        ]
        return cls(
            layout=layout,
            style=style,
            number_of_bodies=len(world.bodies),
            annotated_bodies=annotated_bodies,
        )

    def render(self) -> str:
        """
        Render the summary as a printable multi-line report.

        :return: The report text.
        """
        header = (
            f"RoboCasa kitchen '{self.layout.name}' / '{self.style.name}': "
            f"{self.number_of_bodies} bodies, "
            f"{len(self.annotated_bodies)} semantic annotations"
        )
        lines = [header, "-" * len(header)]
        lines.extend(str(annotated_body) for annotated_body in self.annotated_bodies)
        return "\n".join(lines)


def _start_rviz_publisher(
    world: World,
    tf_period_seconds: float = 0.1,
    marker_period_seconds: float = 2.0,
) -> Optional[threading.Thread]:
    """
    Start publishing the world to RViz on a background thread and return immediately, so
    the caller can keep driving the world (for example while a robot performs a plan)
    and have that motion show up live rather than only the final state.

    The tf tree is re-published frequently and the marker array less often, both on
    timers. A world that stops changing would otherwise have its tf tree published only
    once and expire from RViz's tf buffer (or be missed entirely if RViz connects
    later), leaving the markers unplaceable. The tf tree is cheap to publish; rebuilding
    the markers is not, hence the two rates.

    :param world: The world to visualize.
    :param tf_period_seconds: How often to re-publish the tf tree.
    :param marker_period_seconds: How often to re-publish the marker array.
    :return: The background spinner thread, or None if ROS 2 is unavailable.
    """
    if rclpy is None:
        logger.warning("ROS 2 (rclpy) not available; skipping RViz visualization.")
        return None

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("robocasa_kitchen_demo")
    marker_publisher = VizMarkerPublisher(_world=world, node=node)
    tf_publisher = TFPublisher(_world=world, node=node)

    node.create_timer(tf_period_seconds, tf_publisher.on_state_change)
    node.create_timer(marker_period_seconds, marker_publisher.on_model_change)

    root_frame = str(world.root.name)
    logger.info(
        "Publishing kitchen to RViz. In RViz set:\n"
        "  - Fixed Frame: %s\n"
        "  - add a MarkerArray display on topic '/semworld/viz_marker' "
        "with Durability Policy 'Transient Local'",
        root_frame,
    )
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    return spinner


@dataclass
class CounterSurface:
    """
    The horizontal top surface of a kitchen counter, expressed in the world frame, that
    an object can be placed on.
    """

    top_z: float
    """
    Height of the counter's top surface.
    """

    accessible_edge_y: float
    """
    The ``y`` coordinate of the counter edge that faces the open floor, where the robot
    stands.
    """

    min_x: float
    """
    Lower ``x`` bound of the counter surface.
    """

    max_x: float
    """
    Upper ``x`` bound of the counter surface.
    """


def _counter_top_surface(world: World, counter: CounterTop) -> CounterSurface:
    """
    Compute the placement surface of a counter from its collision geometry.

    :param world: The world the counter belongs to.
    :param counter: The counter annotation to measure.
    :return: The counter's top surface in the world frame.
    """
    surface_boxes = [
        box
        for box in counter.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_boxes
        if box.max_z < 2.0
    ]
    return CounterSurface(
        top_z=max(box.max_z for box in surface_boxes),
        accessible_edge_y=max(box.max_y for box in surface_boxes),
        min_x=min(box.min_x for box in surface_boxes),
        max_x=max(box.max_x for box in surface_boxes),
    )


def _largest_counter_top(world: World) -> CounterTop:
    """
    Return the counter with the largest footprint, the most convenient one to place an
    object on.

    :param world: The kitchen world to search.
    :return: The counter annotation with the largest footprint.
    :raises LookupError: if the world has no counter.
    """
    counters = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, CounterTop)
    ]
    if not counters:
        raise LookupError("The kitchen has no counter to place an object on.")

    def footprint(counter: CounterTop) -> float:
        """
        Compute the horizontal area a counter occupies in the world frame.

        :param counter: The counter to measure.
        :return: The counter's footprint area.
        """
        bounding_box = counter.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        return bounding_box.width * bounding_box.depth

    return max(counters, key=footprint)


def _spawn_robot_and_prepare_pick_up(
    world: World,
    standoff_distance: float = 0.55,
    edge_inset: float = 0.15,
) -> Callable[[], None]:
    """
    Spawn a PR2 at a kitchen counter with an apple resting on the counter surface, and
    return a callable that performs the pick-up.

    Splitting spawning from performing lets the caller start the RViz publisher in between: the
    robot and object are added to the world first (a one-off topology change), then the returned
    callable runs the plan, so a publisher started in the meantime shows the motion live.

    A free graspable object is used rather than one of RoboCasa's articulated fixtures: opening a
    RoboCasa door or drawer needs joint velocity/acceleration limits that its MJCF does not provide,
    which the motion planner requires.

    .. note::
        The robot is placed facing the counter's :attr:`~CounterSurface.accessible_edge_y` edge,
        assuming that edge faces open floor, which holds for the default layout.

    :param world: The kitchen world to spawn the robot into, modified in place.
    :param standoff_distance: How far in front of the counter edge the robot stands.
    :param edge_inset: How far in from the counter edge the apple is placed.
    :return: A callable that performs the pick-up plan and reports the outcome.
    """
    surface = _counter_top_surface(world, _largest_counter_top(world))

    apple_world = RoboCasaDatasetLoader().load_object(RoboCasaObjectCategory.APPLE)
    apple_body = apple_world.bodies_with_collision[0]
    apple_name = apple_body.name
    apple_half_height = (
        apple_body.collision.as_bounding_box_collection_in_frame(apple_world.root)
        .bounding_box()
        .height
        / 2
    )

    place_x = surface.max_x - 0.55
    apple_y = surface.accessible_edge_y - edge_inset
    apple_z = surface.top_z + apple_half_height
    robot_y = apple_y + standoff_distance
    robot_yaw = -math.pi / 2

    pr2_world = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    with world.modify_world():
        drive = OmniDrive.create_with_dofs(
            parent=world.root, child=pr2_world.root, world=world
        )
        world.merge_world(pr2_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            place_x, robot_y, 0.0, 0.0, 0.0, robot_yaw
        )
        world.merge_world_at_pose(
            apple_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                place_x,
                apple_y,
                apple_z,
                reference_frame=world.root,
            ),
        )
    with world.modify_world():
        world.add_semantic_annotation(Apple(root=world.get_body_by_name(apple_name)))

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.evaluate_conditions = False

    apple = world.get_body_by_name(apple_name)
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            PickUpAction(
                apple,
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    pr2.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    ).plan

    def perform() -> None:
        height_before = world.compute_forward_kinematics(world.root, apple).to_np()[
            2, 3
        ]
        logger.info("Spawned PR2; parking arms, raising torso, picking up an apple ...")
        try:
            with simulated_robot:
                plan.perform()
        except PlanFailure as failure:
            logger.warning("Robot could not complete the pick-up: %s", failure)
            return
        height_after = world.compute_forward_kinematics(world.root, apple).to_np()[2, 3]
        lift = float(height_after - height_before)
        outcome = "PICKED UP" if lift > 0.01 else "not lifted"
        logger.info("Apple lifted by %.3f m -> %s", lift, outcome)

    return perform


def load_kitchen(layout: LayoutType, style: StyleType) -> World:
    """
    Load a RoboCasa kitchen into a CRAM world.

    :param layout: The RoboCasa layout to compose.
    :param style: The RoboCasa visual style to compose.
    :return: The loaded world, with semantic annotations attached by the adapter.
    """
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        raise FileNotFoundError(
            f"RoboCasa assets not found at {loader.directory}. Download them once with "
            "'python -m robocasa.scripts.download_kitchen_assets'."
        )
    return loader.load_kitchen(layout_id=layout, style_id=style)


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting the kitchen layout, style, and whether to
    visualize.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default=LayoutType.LAYOUT001.name,
        choices=[layout.name for layout in LayoutType],
        help="RoboCasa kitchen layout to load.",
    )
    parser.add_argument(
        "--style",
        default=StyleType.STYLE001.name,
        choices=[style.name for style in StyleType],
        help="RoboCasa visual style to load.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Publish the loaded kitchen to RViz (requires ROS 2).",
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Spawn a PR2 at a counter and have it pick up an apple off it (requires coraplex).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Load a RoboCasa kitchen, report a summary of its bodies and semantic annotations,
    optionally spawn a PR2 that picks up an apple off a counter, and optionally publish
    the scene to RViz.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments()
    layout = LayoutType[arguments.layout]
    style = StyleType[arguments.style]

    world = load_kitchen(layout, style)
    logger.info("%s", world.root.name)
    logger.info("%s", KitchenSceneSummary.from_world(world, layout, style).render())

    # Spawn the robot (a one-off topology change) before starting the publisher, then start the
    # publisher, then perform the plan: this way the publisher is already live and shows the robot
    # moving instead of only the final state.
    perform_pick_up = (
        _spawn_robot_and_prepare_pick_up(world) if arguments.robot else None
    )

    spinner = _start_rviz_publisher(world) if arguments.visualize else None

    if perform_pick_up is not None:
        perform_pick_up()

    if spinner is not None:
        logger.info("Press Ctrl+C to stop.")
        try:
            spinner.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
