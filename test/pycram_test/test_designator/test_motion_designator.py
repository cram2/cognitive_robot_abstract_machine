import os
import unittest
from copy import deepcopy

import numpy as np

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    Arms,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.plan import MotionNode
from pycram.process_module import simulated_robot, no_execution, real_robot
from pycram.robot_plans import (
    MoveMotion,
    BaseMotion,
    PickUpActionDescription,
    NavigateActionDescription,
    MoveTorsoActionDescription,
    PickUpAction,
)
from pycram.testing import ApartmentWorldTestCase, EmptyWorldTestCase
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2

try:
    from pycram.alternative_motion_mappings.hsrb_motion_mapping import *

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True


@unittest.skipIf(skip_tests, "Alternative motion mappings not available")
class TestActionDesignatorGrounding(ApartmentWorldTestCase):

    def test_pick_up_motion(self):
        test_world = deepcopy(self.world)
        test_robot = PR2.from_world(test_world)
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            test_robot.left_arm.manipulator,
        )
        description = PickUpActionDescription(
            test_world.get_body_by_name("milk.stl"), [Arms.LEFT], [grasp_description]
        )

        plan = plan = SequentialPlan(
            Context.from_world(test_world),
            NavigateActionDescription(
                PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], test_world.root),
                True,
            ),
            MoveTorsoActionDescription([TorsoState.HIGH]),
            description,
        )
        with simulated_robot:
            plan.perform()

        pick_up_node = plan.get_nodes_by_designator_type(PickUpAction)[0]

        motion_nodes = list(
            filter(lambda x: isinstance(x, MotionNode), pick_up_node.recursive_children)
        )

        self.assertEqual(len(motion_nodes), 5)

        motion_charts = [type(m.designator_ref.motion_chart) for m in motion_nodes]
        self.assertTrue(all(mc is not None for mc in motion_charts))
        self.assertIn(CartesianPose, motion_charts)
        self.assertIn(JointPositionList, motion_charts)

    def test_move_motion_chart(self):
        motion = MoveMotion(PoseStamped.from_list([1, 1, 1], frame=self.world.root))
        SequentialPlan(self.context, motion)

        msc = motion.motion_chart

        self.assertTrue(msc)
        np.testing.assert_equal(
            msc.goal_pose.to_position().to_np(), np.array([1, 1, 1, 1])
        )


@unittest.skipIf(skip_tests, "Alternative motion mappings not available")
class TestAlternativeMotionMapping(EmptyWorldTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        hsrb = URDFParser.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "pycram",
                "resources",
                "robots",
                "hsrb.urdf",
            )
        ).parse()

        cls.world.merge_world(hsrb)

        cls.hsr_view = HSRB.from_world(cls.world)
        cls.hsr_context = Context(cls.world, cls.hsr_view)

    def test_alternative_mapping(self):
        move_motion = MoveMotion(
            PoseStamped.from_list([1, 1, 1], frame=self.world.root)
        )

        plan = SequentialPlan(self.hsr_context, move_motion)

        with real_robot:
            self.assertTrue(move_motion.get_alternative_motion())
            msc = move_motion.motion_chart
            self.assertEqual(NavigateActionServerTask, type(msc))
