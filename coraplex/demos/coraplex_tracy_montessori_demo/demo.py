"""
Tracy sorting the Montessori pieces from its table into the holes of the shape-sorting
board.

The scene and the plan are the same however the demo is carried out; only the execution
type differs. Choose it at the bottom of this file, then run the file from its own
folder::

    python demo.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from montessori_board import BoardGeometry, HoleShape, board_body
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    MontessoriPiece,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
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

# %% when the demo cannot run


@dataclass
class DemoCannotBeCarriedOut(Exception):
    """
    Raised when the demo is asked to run with an execution type it has no runner for.
    """

    execution_type: ExecutionType
    """
    The execution type asked for.
    """

    def __str__(self) -> str:
        return f"The demo cannot be carried out as {self.execution_type.name}."


# %% the scene the demo runs in


@dataclass
class MontessoriEnvironment:
    """
    Tracy's table, with the shape-sorting board standing on it and a row of loose pieces
    in front of the board, each to be sorted into the hole it fits through.
    """

    table_top_z: float = 0.88
    """
    Height of the work surface of the table Tracy is bolted to, which every piece and the
    board rest on. It is the top of the table's own collision slab, read off the robot's
    description rather than guessed.
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
    How far apart the loose pieces stand, so reaching for one does not sweep the arm into
    its neighbour.
    """

    release_height: float = 0.01
    """
    How far above the board's lid a piece is let go of, so it drops into its hole rather
    than being pushed against the lid.
    """

    resting_clearance: float = 0.001
    """
    Gap left between a piece and what it stands on, so it rests rather than intersects.
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

    finger_pad_closing_travel: float = 0.0135
    """
    How far Tracy's finger pads move out along the gripper's reach as it closes.

    Measured on the robot's own description: open, the pads end at the tool frame;
    closed, 13.5 mm past it. The Robotiq linkage swings the pads down onto whatever the
    hand is lowered over.
    """

    closed_pad_clearance: float = 0.005
    """
    How far the closed finger pads stay above what a grasped piece stands on.
    """

    solid_contact_stiffness: ContactStiffness = field(
        default_factory=lambda: ContactStiffness(
            time_constant=timedelta(milliseconds=2), damping_ratio=1.0
        )
    )
    """
    How stiffly the solid bodies of the scene push back once something touches them: as
    stiffly as a physics engine stepping every millisecond stays stable at, a time
    constant of two steps.

    The wooden pieces, the board and the robot are solid, so nothing may sink into them;
    MuJoCo's default of 20 ms lets a firmly held piece sink several millimetres into the
    pads.
    """

    solid_contact_impedance: ContactImpedance = field(
        default_factory=lambda: ContactImpedance(
            minimum=0.99, maximum=0.999, width=0.001
        )
    )
    """
    How little the solid bodies of the scene give way under load: nearly not at all.
    """

    board_geometry: BoardGeometry = field(default_factory=BoardGeometry.from_mesh)
    """
    The board's holes and lid, read off its mesh.
    """

    board: Body = field(init=False)
    """
    The board, once :meth:`populate` has stood it on the table.
    """

    holes_of_pieces: Dict[MontessoriPiece, HoleShape] = field(
        init=False, default_factory=dict
    )
    """
    The hole each loose piece is sorted into, in the order the pieces are sorted.
    """

    @property
    def grasp_height(self) -> float:
        """
        How far above its bottom face a piece is gripped.

        A grasp brings the tool frame to the grasp point, and the pads end there while
        the hand is open; closing, they travel :attr:`finger_pad_closing_travel` further
        down. Gripped this high, the closed pads stay :attr:`closed_pad_clearance` above
        the table. Gripped at its middle instead, the closing pads would reach below the
        piece's bottom face and press into the table.
        """
        return self.finger_pad_closing_travel + self.closed_pad_clearance

    def piece_shapes(self) -> Dict[HoleShape, Shape]:
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

    def get_world(self) -> World:
        """
        Create a world holding Tracy, read from its own description, which also equips
        its arms and grippers with the servos a physical simulation needs, with its arms
        parked and its body as solid as the pieces it holds, and stand the board and the
        pieces in it.

        A contact is as soft as the softer of the two shapes in it, so a solid piece held
        between soft pads would still sink into them.
        """
        world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
        robot = Tracy.from_world(world)
        self.solid(ContactFriction()).apply_to(world.bodies_with_collision)
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        self.populate(world)
        return world

    def solid(self, friction: ContactFriction) -> ContactParameters:
        """
        :param friction: The friction the surface offers.
        :return: The contact of a solid surface offering ``friction``.
        """
        return ContactParameters(
            friction=friction,
            stiffness=self.solid_contact_stiffness,
            impedance=self.solid_contact_impedance,
        )

    def populate(self, world: World) -> None:
        """
        Stand the shape-sorting board and the loose pieces on Tracy's table.

        The board is bolted down, while every piece hangs off a connection with six
        degrees of freedom, which is what lets the robot take it somewhere else.

        :param world: The world holding Tracy.
        """
        with world.modify_world():
            self._stand_the_board(world)
            for index, (hole, shape) in enumerate(self.piece_shapes().items()):
                piece = self._stand_piece(
                    world,
                    PrefixedName(f"{hole}_piece"),
                    shape,
                    self.piece_row_start_y + index * self.piece_row_spacing,
                )
                self.holes_of_pieces[piece] = hole

    def build_plan(self, context: Context) -> Plan:
        """
        Pick every piece up in turn and let it go over the hole it fits through.

        The plan begins by parking the arms: a simulated Tracy starts parked already, the
        real one wherever it was left.

        :param context: The context the actions are built in.
        :return: The plan that sorts the pieces.
        """
        robot = context.robot
        actions = [ParkArmsAction(robot.all_arms)]
        for piece, hole in self.holes_of_pieces.items():
            [grasp] = piece.grasp_candidates()
            actions.append(PickUpAction(grasp, robot.left_arm))
            actions.append(
                PlaceAction(piece, self.release_pose_over(hole, context.world))
            )
        actions.append(ParkArmsAction(robot.all_arms))
        return sequential(actions, context=context).plan

    def release_pose_over(self, hole: HoleShape, world: World) -> Pose:
        """
        :param hole: The hole a piece is sorted into.
        :param world: The world the board stands in.
        :return: Where that piece is let go of: straight above the hole and turned the
            way it is, :attr:`release_height` above the lid.
        """
        hole_x, hole_y = self.board_geometry.hole_of_shape(hole).bounding_box_center
        board_T_release = Pose.from_xyz_rpy(
            hole_x,
            hole_y,
            self.board_geometry.lid_top + self.release_height + self.piece_height / 2,
            reference_frame=self.board,
        )
        return world.transform(board_T_release, world.root)

    def _stand_the_board(self, world: World) -> None:
        """
        Bolt the board to the table straight in front of Tracy.

        :param world: The world being built, already open for modification.
        """
        self.board = board_body(
            PrefixedName("shape_sorting_board"), self.board_geometry
        )
        self.solid(ContactFriction()).apply_to([self.board])
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
                    self.table_top_z - self.board_geometry.bottom,
                ),
            )
        )

    def _stand_piece(
        self, world: World, name: PrefixedName, shape: Shape, row_y: float
    ) -> MontessoriPiece:
        """
        Stand one loose piece in the row in front of Tracy.

        :param world: The world being built, already open for modification.
        :param name: The name the piece's body carries.
        :param shape: The piece's geometry, centred on its own origin.
        :param row_y: Where along the row it stands.
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
                    self.table_top_z + self.resting_clearance + self.piece_height / 2,
                ),
            )
        )
        self.solid(ContactParameters.create_for_grasped_object().friction).apply_to(
            [body]
        )
        piece = MontessoriPiece(root=body, grasp_height=self.grasp_height)
        world.add_semantic_annotations([piece])
        return piece


# %% running it


def main(execution_type: ExecutionType) -> None:
    """
    Carry the demo out the way ``execution_type`` names.

    Each runner is imported only once it is chosen, so a run needs nothing installed for
    the execution types it does not use.

    :param execution_type: How the plan is carried out.
    """
    environment = MontessoriEnvironment()
    match execution_type:
        case ExecutionType.PHYSICALLY_SIMULATED:
            import mujoco_demo

            mujoco_demo.run(environment, environment.get_world())
        case ExecutionType.KINEMATICALLY_SIMULATED:
            import rviz_demo

            rviz_demo.run(environment, environment.get_world())
        case ExecutionType.REAL:
            import real_demo

            with real_demo.running_robot() as robot:
                environment.populate(robot.world)
                real_demo.run(environment, robot)
        case _:
            raise DemoCannotBeCarriedOut(execution_type)


if __name__ == "__main__":
    main(ExecutionType.KINEMATICALLY_SIMULATED)
