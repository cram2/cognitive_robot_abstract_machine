"""
A PR2 takes a milk out of the small kitchen's fridge and puts it down on the kitchen
island.

The plan states the transport only. That the milk stands behind a closed fridge door is
not written into it: the container the milk is in is found from the milk, and the steps
that open it and shut it again are wrapped around the transport.

The world is built from specifications: the kitchen comes from a URDF, the robot from a
:class:`~semantic_digital_twin.api.RobotSpecification`, and the shelf and the milk from
body and annotation specifications. Building the world furnishes the kitchen; populating
the scene puts down the milk the plan is about, and nothing else.

..note:: This demonstration is written for a simulated run. Running it with
    :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` fetches the world from a
    controller, and the world fetcher does not run
    :class:`~semantic_digital_twin.reasoning.world_reasoner.WorldReasoner` over what it
    receives, so the fridge annotation the scene is placed against would be missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing_extensions import ClassVar, List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import giskard_reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionLike, PlanNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a, an, entity, the, variable
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, IsStorageSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Fridge,
    Milk,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% what holds what


@symbolic_function
def contains(
    container: KinematicStructureEntity, body: KinematicStructureEntity
) -> bool:
    """
    :param container: The entity that may hold the body.
    :param body: The entity that may be held.
    :return: Whether nearly all of the body lies inside the container.
    """
    return InsideOf(body, container)() > 0.9


# %% the demonstration


@dataclass
class KitchenFridgeDemonstration(RobotDemonstration):
    """
    A robot fetches a milk out of the kitchen fridge and puts it on the kitchen island.
    """

    ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"

    shelf_layer_name: str = "fridge_shelf"
    """
    Name of the shelf layer the milk stands on.
    """

    milk_arm: Arms = Arms.RIGHT
    """
    The arm carrying the milk.
    """

    milk_mesh_path: Path = (
        Path(__file__).parents[2] / "resources" / "objects" / "milk.stl"
    )
    """
    The mesh the milk carton is spawned from.
    """

    def build_simulated_world(self) -> World:
        """
        Load the kitchen from its URDF, put the robot in it, infer what its bodies mean.

        -- which is what turns the fridge's parts into a fridge with a door and a handle
        -- and stand a shelf layer in the fridge.
        """
        world = WorldSpecification.from_urdf(
            str(
                Path(__file__).parents[2]
                / "resources"
                / "worlds"
                / "kitchen-small.urdf"
            ),
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=HomogeneousTransformationMatrix(),
                )
            ],
        ).to_domain_object()

        with world.modify_world():
            WorldReasoner(world).reason()

        fridge = variable(Fridge, domain=world.semantic_annotations)
        fridge_annotation = the(entity(fridge)).first()
        shelf_layer = ShelfLayer.get_annotation_specification(
            self.shelf_layer_name,
            BodySpecification.box(
                self.shelf_layer_name,
                Scale(0.45, 0.5, 0.02),
                color=Color(0.9, 0.93, 0.95),
            ),
        ).spawn(
            world,
            parent=fridge_annotation.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.0, 0.02, 0.0, yaw=np.pi
            ),
        )
        with world.modify_world():
            fridge_annotation.add(shelf_layer)

        return world

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def get_milk_specification(self) -> SemanticAnnotationWithRootSpecification[Milk]:
        """
        Describe the milk carton, whose shape is the mesh it is spawned from.

        The mesh's own origin lies below the carton's centre, while a body is placed,
        and a spot on a surface sampled for it, by measuring equal halves out from its
        frame. The geometry is therefore centred on that frame here, once, rather than
        every pose being offset against it.

        :return: The specification the milk is spawned from.
        """
        name = "milk"
        carton = Mesh(filename=str(self.milk_mesh_path))
        return Milk.get_annotation_specification(
            name,
            BodySpecification.mesh(
                name,
                carton.filename,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *-carton.mesh.bounds.mean(axis=0)
                ),
            ),
        )

    def populate_scene(self, world: World) -> None:
        """
        Stand the milk on the fridge's shelf layer.
        """
        specification = self.get_milk_specification()

        shelf_layer = variable(ShelfLayer, domain=world.semantic_annotations)
        shelf_layer_body = (
            the(
                entity(shelf_layer).where(
                    shelf_layer.name.name == self.shelf_layer_name
                )
            )
            .first()
            .root
        )
        specification.spawn(
            world,
            parent=shelf_layer_body,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.16,
                0.0,
                (
                    shelf_layer_body.collision.scale.z
                    + specification.root_specification.scale.z
                )
                / 2,
            ),
        )

    @staticmethod
    def island_counter_top(world: World) -> CounterTop:
        """
        :param world: The world holding the kitchen.
        :return: The counter top the milk is put down on, told apart from the sink's by
            the body the kitchen names it after.
        """
        counter_top = variable(CounterTop, domain=world.semantic_annotations)
        return the(
            entity(counter_top).where(
                counter_top.root.name.name == "kitchen_island_surface"
            )
        ).first()

    def place_spot_on_island(self, world: World, milk: Milk) -> Point3:
        """
        Pick a spot on the kitchen island to put the milk down on.

        The counter top works out where its own surface is and how high the milk has to
        stand to rest on it, so the spot is drawn from the surface rather than measured
        off a bounding box. It comes out somewhere different on every run.

        A spot, not a pose: which way the carton ends up turned is nobody's business
        here. The robot walks round the island to wherever the spot is, and the heading
        follows from where it ends up standing.

        :param world: The world holding the kitchen.
        :param milk: The carton the spot has to be big enough for.
        :return: The spot, in the world frame.
        """
        counter_top = self.island_counter_top(world)
        points = counter_top.sample_points_from_surface(
            body_to_sample_for=milk, amount=100
        )
        return world.transform(next(iter(points)), world.root)

    @staticmethod
    def pose_the_robot_faces(spot: Point3, context: Context) -> Pose:
        """
        :param spot: Where the object has to end up.
        :param context: The context holding the robot.
        :return: That spot, turned the way the robot is standing.
        """
        return Pose.from_xyz_rpy(
            spot.x,
            spot.y,
            spot.z,
            yaw=context.robot.root.global_pose.yaw,
            reference_frame=context.world.root,
        )

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the robot in ``world``.

        Conditions are evaluated, which is what makes the underspecified arms work: a
        hand that cannot reach the milk, or that is not the one holding it, fails its
        precondition before it moves and the next candidate is tried.
        """
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        context = Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=self.ros_node,
            evaluate_conditions=True,
        )
        context.debug = True
        return context

    # %% reaching into containers

    @staticmethod
    def handle_of_enclosing_container(body: Body, world: World) -> Optional[Body]:
        """
        Find what has to be pulled open before ``body`` can be taken.

        Asks for the storage spaces whose own body holds the given one and takes the
        handle of the first door among them. In this kitchen the milk is held by both the
        fridge and the rack the fridge stands in, and only the fridge has a door.

        ..note:: Only doors are followed. A drawer holds what it contains in its own case
            and is found just as well, but its handle hangs off the drawer itself rather
            than off a door.

        :param body: The body that may stand inside a container.
        :param world: The world holding both.
        :return: The handle of the container, or ``None`` when the body stands in the
            open.
        """
        storage_space = variable(IsStorageSpace, domain=world.semantic_annotations)
        containers = an(
            entity(storage_space).where(contains(storage_space.root, body))
        ).evaluate()
        return next(
            (
                container.doors[0].handle.root
                for container in containers
                if isinstance(container, HasDoors) and container.doors
            ),
            None,
        )

    def add_container_opening_and_closing(
        self, actions: List[ActionLike], body: Body, context: Context
    ) -> List[ActionLike]:
        """
        Open the container ``body`` stands in before the given steps, and shut it again
        after them.

        Which hand does the pulling is left to the plan, and so is the side the gripper
        comes at the handle from; the standing pose is looked for with a named hand.

        :param actions: The steps that need the container open.
        :param body: The body those steps reach for.
        :param context: The context the standing poses are chosen in.
        :return: The steps with the container steps around them, and ``actions`` itself
            when the body stands in the open.
        """
        handle = self.handle_of_enclosing_container(body, context.world)
        if handle is None:
            return actions

        door_arm = Arms.LEFT

        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(handle, context, door_arm)
                    ),
                ),
            ),
            a(OpenAction)(
                object_designator=handle,
                arm=...,
                goal_joint_state=1.45,
            ),
            ParkArmsAction(Arms.BOTH),
            *actions,
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(handle, context, door_arm)
                    ),
                ),
            ),
            a(CloseAction)(
                object_designator=handle,
                arm=...,
                goal_joint_state=0.0,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

    # %% the plan

    def get_grasp_for_milk(self, context: Context) -> Match[GraspDescription]:
        """
        Describe a grasp without saying which side of the object the gripper comes from.

        The plan settles that once it is standing in front of the object, rather than
        against the pose the robot happened to start the plan in.

        :param context: The context holding the robot that reaches for it.
        :return: The grasp, still open on its approach direction.
        """
        return a(GraspDescription)(
            approach_direction=...,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=ViewManager.get_end_effector_view(
                self.milk_arm, context.robot
            ),
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the milk to the kitchen island, opening and shutting whatever it stands
        in.

        Every action of a plan is expanded before its first motion runs, which is too
        early to choose a standing pose: earlier steps still move both the robot and the
        handle riding the swinging door. The navigations are therefore underspecified and
        take a deferred location, so a standing pose is chosen once execution reaches
        them.

        The pick-up leaves the side it grasps from open, and the place the heading it
        puts the carton down at, so both are settled against the world the plan finds
        rather than against the pose the robot started in.
        """
        world = context.world
        milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
        milk_body = milk.root
        place_spot = self.place_spot_on_island(world, milk)

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            milk_body, context, self.milk_arm
                        )
                    ),
                ),
            ),
            a(PickUpAction)(
                object_designator=milk_body,
                grasp_description=self.get_grasp_for_milk(context),
                arm=self.milk_arm,
            ),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            place_spot, context, self.milk_arm
                        )
                    ),
                ),
            ),
            a(PlaceAction)(
                object_designator=milk_body,
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: [self.pose_the_robot_faces(place_spot, context)]
                    ),
                ),
                arm=self.milk_arm,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                *self.add_container_opening_and_closing(transport, milk_body, context),
            ],
            context,
        )


# %% running the demo


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> World:
    """
    Run the demonstration and check that the milk ended up on the kitchen island with
    the fridge shut behind it.

    :param execution_type: Whether to drive the real robot or simulate it.
    :return: The world the demonstration acted on.
    """
    demonstration = KitchenFridgeDemonstration(
        used_robot=PR2, execution_type=execution_type
    )
    world = demonstration.run()

    milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
    milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    counter_top = demonstration.island_counter_top(world)
    surface_box = (
        counter_top.supporting_surface.area.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
    )
    fridge = the(entity(variable(Fridge, domain=world.semantic_annotations))).first()
    door_angle = fridge.doors[0].root.parent_connection.position

    print(f"milk placed at {np.round(milk.root.global_pose.to_position(), 3)}")
    print(
        f"it rests at {milk_box.min_z:.4f}, the island's top is {surface_box.max_z:.4f}"
    )
    print(f"fridge door closed to {door_angle:.4f} rad")

    placement_tolerance = 0.02
    assert abs(milk_box.min_z - surface_box.max_z) < placement_tolerance
    assert surface_box.min_x <= milk_box.min_x and milk_box.max_x <= surface_box.max_x
    assert surface_box.min_y <= milk_box.min_y and milk_box.max_y <= surface_box.max_y
    assert door_angle < 0.02
    return world


if __name__ == "__main__":
    main()
