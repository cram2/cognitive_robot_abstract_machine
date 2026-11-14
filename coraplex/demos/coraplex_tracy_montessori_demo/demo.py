"""
Tracy sorting the Montessori pieces from its table into the holes of the shape-sorting
board.

The scene and the plan are the same however the demo is carried out; only the runner
differs. Run the runner for the execution type wanted from this folder::

    python mujoco_demo.py  # physically simulated, in a MuJoCo viewer
    python rviz_demo.py    # kinematically simulated, drawn in RViz
    python real_demo.py    # on the real robot
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from montessori_board import ShapeSortingBoardModel
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
    ShapeSortingBoard,
    ShapeSortingHole,
    ShapeSortingPiece,
)
from semantic_digital_twin.shape_sorting_toy.pieces import PieceSet
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
)
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% solid bodies


@dataclass
class SolidContact:
    """
    How the solid bodies of the scene push back once something touches them.

    The wooden pieces, the board and the robot are solid, so nothing may sink into them;
    MuJoCo's soft defaults let a firmly held piece sink several millimetres into the
    pads.
    """

    stiffness: ContactStiffness = field(
        default_factory=lambda: ContactStiffness(
            time_constant=timedelta(milliseconds=2), damping_ratio=1.0
        )
    )
    """
    As stiff as a physics engine stepping every millisecond stays stable at: a time
    constant of two steps, where MuJoCo's default is 20 ms.
    """

    impedance: ContactImpedance = field(
        default_factory=lambda: ContactImpedance(
            minimum=0.99, maximum=0.999, width=0.001
        )
    )
    """
    How little the solid bodies give way under load: nearly not at all.
    """

    def with_friction(self, friction: ContactFriction) -> ContactParameters:
        """
        :param friction: The friction the surface offers.
        :return: The contact of a solid surface offering ``friction``.
        """
        return ContactParameters(
            friction=friction, stiffness=self.stiffness, impedance=self.impedance
        )


# %% gripping a piece


@dataclass
class PieceGrip:
    """
    How low on a piece the sorting gripper takes hold of it.
    """

    closed_pad_clearance: float = 0.005
    """
    How far the closed finger pads stay above what a gripped piece stands on.
    """

    def grasp_height(self, gripper: HasTwoFingers) -> float:
        """
        A grasp is aimed at where the closed fingers meet.

        The pads reach further along the gripper than that point: open, they end at the tool frame, as on Tracy's
        Robotiq gripper, and closing carries them
        :attr:`~semantic_digital_twin.robots.robot_part_mixins.HasTwoFingers.travel_when_closing`
        further. Gripped this high, the closed pads stay :attr:`closed_pad_clearance`
        above what the piece stands on, and the piece hangs as little as possible below
        them when it is let go of over its hole.

        :param gripper: The gripper that takes hold of the pieces.
        :return: How far above its bottom face a piece is gripped, at the point the
            closed fingers meet.
        """
        tool_frame_above_the_meeting_point = -float(
            gripper.tool_P_grasp_when_closed.vector3.dot(gripper.approach_axis)
        )
        return (
            self.closed_pad_clearance
            + gripper.travel_when_closing
            + tool_frame_above_the_meeting_point
        )


# %% the scene


@dataclass
class MontessoriScene:
    """
    Tracy's table, with the shape-sorting board standing on it and a row of loose pieces
    in front of the board, each to be sorted into the hole it fits through.
    """

    table_top_z: float = 0.88
    """
    Height of the work surface of the table Tracy is bolted to, which every piece and
    the board rest on.

    It is the top of the table's own collision slab, read off the robot's description
    rather than guessed.
    """

    board_x: float = 0.85
    """
    How far in front of Tracy the shape-sorting board stands, straight ahead of it, with
    its drawers facing the robot.
    """

    board_y: float = 0.0
    """
    Where the board stands sideways.
    """

    piece_row_x: float = 0.55
    """
    How far in front of Tracy the loose pieces stand: nearer than the board, and far
    enough from it that reaching for a piece does not sweep the arm through the drawer
    handles.
    """

    piece_row_start_y: float = 0.1
    """
    Where the first loose piece stands sideways, on the side of the sorting arm.
    """

    piece_row_spacing: float = 0.1
    """
    How far apart the loose pieces stand, so reaching for one does not sweep the arm
    into its neighbour.
    """

    resting_clearance: float = 0.001
    """
    Gap left between a piece and what it stands on, so it rests rather than intersects.
    """

    piece_set: PieceSet = field(init=False)
    """
    The loose pieces, measured off the real set, once :meth:`populate` has built them to
    be gripped by the sorting arm.
    """

    solid_contact: SolidContact = field(default_factory=SolidContact)
    """
    How the robot, the board and the pieces push back once touched.
    """

    grip: PieceGrip = field(default_factory=PieceGrip)
    """
    How low on a piece the sorting gripper takes hold of it.
    """

    board_model: ShapeSortingBoardModel = field(default_factory=ShapeSortingBoardModel)
    """
    The board, as read off its mesh, and the board it becomes.
    """

    board: ShapeSortingBoard = field(init=False)
    """
    The board with its holes, once :meth:`populate` has stood it on the table.
    """

    pieces: list[ShapeSortingPiece] = field(init=False, default_factory=list)
    """
    The loose pieces, in the order they are sorted.
    """

    @staticmethod
    def sorting_arm(tracy: Tracy) -> Arm:
        """
        :param tracy: The robot.
        :return: The arm that sorts the pieces: the left one, as in every Tracy demo in
            this repository.
        """
        return tracy.left_arm

    def build_world(self) -> World:
        """
        Create a world holding Tracy, read from its own description, which also equips
        its arms and grippers with the servos a physical simulation needs, with its arms
        parked and its body as solid as the pieces it holds, and stand the board and the
        pieces in it.

        A contact is as soft as the softer of the two shapes in it, so a solid piece
        held between soft pads would still sink into them.

        :return: The world.
        """
        world = URDFParser.from_file(Tracy.get_ros_file_path()).create_world()
        tracy = Tracy.from_world(world)
        self.solid_contact.with_friction(ContactFriction()).apply_to(
            world.bodies_with_collision
        )
        for arm in tracy.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        self.populate(world)
        return world

    def populate(self, world: World) -> None:
        """
        Stand the shape-sorting board and the loose pieces on Tracy's table.

        The board is bolted down, while every piece hangs off a connection with six
        degrees of freedom, which is what lets the robot take it somewhere else.

        :param world: The world holding Tracy.
        """
        [tracy] = world.get_semantic_annotations_by_type(Tracy)
        self.piece_set = PieceSet.full_size(
            grasp_height=self.grip.grasp_height(self.sorting_arm(tracy).end_effector)
        )
        with world.modify_world():
            self._stand_the_board(world)
            for index, measured_piece in enumerate(self.piece_set.pieces):
                piece = self._stand_piece(
                    world=world,
                    measured_piece=measured_piece,
                    hole=self.board.hole_for(piece=measured_piece),
                    row_y=self.piece_row_start_y + index * self.piece_row_spacing,
                )
                self.pieces.append(piece)

    def _stand_the_board(self, world: World) -> None:
        """
        Bolt the board to the table straight in front of Tracy.

        :param world: The world being built, already open for modification.
        """
        self.board = self.board_model.specification(name="shape_sorting_board").spawn(
            world,
            parent=world.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.board_x,
                self.board_y,
                self.table_top_z - self.board_model.mesh.bottom,
            ),
        )
        self.solid_contact.with_friction(ContactFriction()).apply_to([self.board.root])

    def _stand_piece(
        self,
        *,
        world: World,
        measured_piece: ShapeSortingPiece,
        hole: ShapeSortingHole,
        row_y: float,
    ) -> ShapeSortingPiece:
        """
        Stand a copy of one piece of the set in the row in front of Tracy, turned so
        that its footprint follows the hole it is sorted into.

        :param world: The world being built, already open for modification.
        :param measured_piece: The piece of the set to stand a copy of.
        :param hole: The hole the piece is sorted into.
        :param row_y: Where along the row it stands.
        :return: The piece.
        """
        [shape] = measured_piece.root.collision.shapes
        body = Body(name=measured_piece.name)
        body.collision = ShapeCollection(
            [
                dataclasses.replace(
                    shape, origin=HomogeneousTransformationMatrix(reference_frame=body)
                )
            ],
            reference_frame=body,
        )
        body.visual = ShapeCollection(
            [
                dataclasses.replace(
                    shape, origin=HomogeneousTransformationMatrix(reference_frame=body)
                )
            ],
            reference_frame=body,
        )
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=body,
                name=PrefixedName(f"{body.name.name}_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.piece_row_x,
                    row_y,
                    self.table_top_z
                    + self.resting_clearance
                    + measured_piece.height / 2,
                    yaw=measured_piece.turn_onto(hole=hole),
                ),
            )
        )
        self.solid_contact.with_friction(
            ContactParameters.create_for_grasped_object().friction
        ).apply_to([body])
        piece = type(measured_piece)(
            root=body, grasp_height=measured_piece.grasp_height
        )
        world.add_semantic_annotations([piece])
        return piece


# %% the plan


@dataclass
class ShapeSortingPlan:
    """
    The plan that picks every piece of a scene up in turn and lets it go over the hole
    it fits through.
    """

    scene: MontessoriScene
    """
    The scene whose pieces are sorted.
    """

    release_height: float = 0.01
    """
    How far above the board's lid a piece is let go of, so it drops into its hole rather
    than being pushed against the lid.
    """

    def build(self, context: StatechartContext) -> Statechart:
        """
        The plan begins by parking the arms: a simulated Tracy starts parked already,
        the real one wherever it was left.

        :param context: The context of the executor that performs the plan.
        :return: The statechart that sorts the pieces.
        """
        robot = context.require_extension(extension_type=RobotAccess).robot
        actions = [ParkArmsAction(arms=robot.all_arms)]
        for piece in self.scene.pieces:
            [grasp] = piece.grasp_candidates()
            actions.append(PickUpAction(grasp=grasp, arm=self.scene.sorting_arm(robot)))
            actions.append(
                PlaceAction(
                    object_designator=piece,
                    target_location=self.release_pose_over(
                        piece=piece,
                        hole=self.scene.board.hole_for(piece=piece),
                        world=context.world,
                    ),
                )
            )
        actions.append(ParkArmsAction(arms=robot.all_arms))
        statechart = Statechart(context=context)
        statechart.add_node(node=Sequence(nodes=actions))
        return statechart

    def release_pose_over(
        self, *, piece: ShapeSortingPiece, hole: ShapeSortingHole, world: World
    ) -> Pose:
        """
        :param piece: The piece being sorted.
        :param hole: The hole the piece is sorted into.
        :param world: The world the board stands in.
        :return: Where the piece is let go of: its centroid straight above the hole's,
            :attr:`release_height` above the lid. The piece keeps the orientation it
            stood in, which follows the hole.
        """
        hole_T_release = Pose.from_xyz_rpy(
            0.0,
            0.0,
            self.release_height + piece.height / 2,
            reference_frame=hole.root,
        )
        return world.transform(hole_T_release, world.root)


# %% running it


@dataclass
class DemoRunner(ABC):
    """
    Carries the sorting plan out in one way: physically simulated, kinematically
    simulated, or on the real robot.
    """

    scene: MontessoriScene = field(default_factory=MontessoriScene)
    """
    The scene the pieces are sorted in.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Build the scene, build the plan that sorts its pieces, and carry the plan out.
        """
