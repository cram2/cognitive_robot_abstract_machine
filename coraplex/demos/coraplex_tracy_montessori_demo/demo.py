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

import copy
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
from montessori_board import HoleShape, ShapeSortingBoard
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ShapeSortingPiece,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the pieces


@dataclass
class ShapeSortingPieceSet:
    """
    The loose pieces of the set, with the sizes and colours measured off the real set.
    """

    piece_height: float = 0.03
    """
    How tall every piece of the set stands.
    """

    cube_edge_length: float = 0.03
    """
    Edge length of the cube, which goes through the square hole.
    """

    cylinder_diameter: float = 0.028
    """
    Diameter of the cylinder, which goes through a circular hole.
    """

    rectangular_prism_width: float = 0.02
    """
    Short side of the rectangular prism, which goes through the rectangular hole.
    """

    rectangular_prism_length: float = 0.04
    """
    Long side of the rectangular prism, along y as the rectangular hole's long side is.
    """

    triangular_prism_side: float = 0.037
    """
    Side of the triangular prism's equilateral cross-section, which goes through the
    triangular hole.
    """

    def shapes(self) -> dict[HoleShape, Shape]:
        """
        :return: The shape of the piece that goes through each hole, turned the way that
            hole is, in the order the pieces are sorted.
        """
        return {
            HoleShape.SQUARE: Box(
                scale=Scale(
                    self.cube_edge_length, self.cube_edge_length, self.piece_height
                ),
                color=Color.CYAN(),
            ),
            HoleShape.CIRCLE: Cylinder(
                width=self.cylinder_diameter,
                height=self.piece_height,
                color=Color.CYAN(),
            ),
            HoleShape.RECTANGLE: Box(
                scale=Scale(
                    self.rectangular_prism_width,
                    self.rectangular_prism_length,
                    self.piece_height,
                ),
                color=Color.YELLOW(),
            ),
            HoleShape.TRIANGLE: Mesh.triangular_prism(
                self.triangular_prism_side, self.piece_height
            ),
        }


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

    pieces: ShapeSortingPieceSet = field(default_factory=ShapeSortingPieceSet)
    """
    The loose pieces.
    """

    solid_contact: SolidContact = field(default_factory=SolidContact)
    """
    How the robot, the board and the pieces push back once touched.
    """

    grip: PieceGrip = field(default_factory=PieceGrip)
    """
    How low on a piece the sorting gripper takes hold of it.
    """

    shape_sorting_board: ShapeSortingBoard = field(default_factory=ShapeSortingBoard)
    """
    The board, as read off its mesh, and the body it becomes.
    """

    board: Body = field(init=False)
    """
    The board's body, once :meth:`populate` has stood it on the table.
    """

    holes_of_pieces: dict[ShapeSortingPiece, HoleShape] = field(
        init=False, default_factory=dict
    )
    """
    The hole each loose piece is sorted into, in the order the pieces are sorted.
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
        grasp_height = self.grip.grasp_height(self.sorting_arm(tracy).end_effector)
        with world.modify_world():
            self._stand_the_board(world)
            for index, (hole, shape) in enumerate(self.pieces.shapes().items()):
                piece = self._stand_piece(
                    world,
                    PrefixedName(f"{hole}_piece"),
                    shape,
                    self.piece_row_start_y + index * self.piece_row_spacing,
                    grasp_height,
                )
                self.holes_of_pieces[piece] = hole

    def _stand_the_board(self, world: World) -> None:
        """
        Bolt the board to the table straight in front of Tracy.

        :param world: The world being built, already open for modification.
        """
        self.board = self.shape_sorting_board.body(PrefixedName("shape_sorting_board"))
        self.solid_contact.with_friction(ContactFriction()).apply_to([self.board])
        world.add_kinematic_structure_entity(self.board)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=self.board,
                name=PrefixedName("shape_sorting_board_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.board_x,
                    self.board_y,
                    self.table_top_z - self.shape_sorting_board.geometry.bottom,
                ),
            )
        )

    def _stand_piece(
        self,
        world: World,
        name: PrefixedName,
        shape: Shape,
        row_y: float,
        grasp_height: float,
    ) -> ShapeSortingPiece:
        """
        Stand one loose piece in the row in front of Tracy.

        :param world: The world being built, already open for modification.
        :param name: The name the piece's body carries.
        :param shape: The piece's geometry, centred on its own origin.
        :param row_y: Where along the row it stands.
        :param grasp_height: How far above its bottom face the piece is gripped.
        :return: The piece.
        """
        body = Body(name=name)
        shape.origin = HomogeneousTransformationMatrix(reference_frame=body)
        body.collision = ShapeCollection([shape], reference_frame=body)
        body.visual = ShapeCollection([copy.deepcopy(shape)], reference_frame=body)
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=body,
                name=PrefixedName(f"{name.name}_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.piece_row_x,
                    row_y,
                    self.table_top_z
                    + self.resting_clearance
                    + self.pieces.piece_height / 2,
                ),
            )
        )
        self.solid_contact.with_friction(
            ContactParameters.create_for_grasped_object().friction
        ).apply_to([body])
        piece = ShapeSortingPiece(root=body, grasp_height=grasp_height)
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
        robot = context.require_extension(RobotAccess).robot
        actions = [ParkArmsAction(arms=robot.all_arms)]
        for piece, hole in self.scene.holes_of_pieces.items():
            [grasp] = piece.grasp_candidates()
            actions.append(PickUpAction(grasp=grasp, arm=self.scene.sorting_arm(robot)))
            actions.append(
                PlaceAction(
                    object_designator=piece,
                    target_location=self.release_pose_over(hole, context.world),
                )
            )
        actions.append(ParkArmsAction(arms=robot.all_arms))
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=actions))
        return statechart

    def release_pose_over(self, hole: HoleShape, world: World) -> Pose:
        """
        :param hole: The hole a piece is sorted into.
        :param world: The world the board stands in.
        :return: Where that piece is let go of: straight above the hole,
            :attr:`release_height` above the lid. The piece keeps the orientation it
            stood in, which is the hole's.
        """
        center = self.scene.shape_sorting_board.geometry.hole_of_shape(
            hole
        ).bounds.center
        board_T_release = Pose.from_xyz_rpy(
            center.x,
            center.y,
            self.scene.shape_sorting_board.geometry.lid_top
            + self.release_height
            + self.scene.pieces.piece_height / 2,
            reference_frame=self.scene.board,
        )
        return world.transform(board_T_release, world.root)


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
