import sys
import time
import unittest
from os.path import dirname
from typing import Optional

from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans import PickUpActionDescription, MoveTorsoActionDescription
from rclpy.logging import set_logger_level
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.definitions import TorsoState
#from pycram.ros import set_logger_level
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Agent, Body

from segmind.segmenters.cram_segmenter import CRAMSegmenter
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector
from pycram.datastructures.enums import Arms, Grasp, WorldMode, LoggerLevel
#from pycram.datastructures.pose import Pose
from pycram.designator import ObjectDesignatorDescription
from pycram.process_module import simulated_robot, ProcessModule
from pycram.plan import Plan

try:
    from pyqt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    QApplication = None
    RDRCaseViewer = None


class TestCRAMPlayer(unittest.TestCase):
    cram_segmenter: CRAMSegmenter
    robot: Agent
    milk: Body
    kitchen: Body
    render_mode: WorldMode = WorldMode.DIRECT
    viz_marker_publisher: VizMarkerPublisher
    world: World
    app: Optional[QApplication] = None
    viewer: Optional[RDRCaseViewer] = None
    use_gui: bool = False

    @classmethod
    def setUpClass(cls):
        #RobotDescriptionManager().load_description("pr2")
        cls.world = World()
        set_logger_level(LoggerLevel.DEBUG)
        cls.viz_marker_publisher = VizMarkerPublisher(world=cls.world)
        cls.cram_segmenter = CRAMSegmenter(cls.world, [GeneralPickUpDetector], plot_timeline=True,
                                           plot_save_path=f"{dirname(__file__)}/test_results/cram_segmenter_test")
        cls.kitchen = Body("kitchen", Kitchen, "kitchen.urdf")
        cls.robot = Body("pr2", Robot, "pr2.urdf", pose=PoseStamped(Pose(Vector3(0.6, 0.4, 0))))
        cls.milk = Body("milk", Milk, "milk.stl", pose=PoseStamped(Pose(Vector3(1.3, 1, 0.9))))
        if cls.use_gui and QApplication is not None:
            cls.app = QApplication(sys.argv)
            cls.viewer = RDRCaseViewer()


    def tearDown(self):
        if Plan.current_plan is not None:
            Plan.current_plan.clear()
        time.sleep(0.05)
        self.world.reset_world(remove_saved_states=True)


    @classmethod
    def tearDownClass(cls):
        GeneralPickUpDetector.start_condition_rdr.save()
        cls.viz_marker_publisher._stop_publishing()
        cls.world.exit()

    def test_pick_up(self):
        # self.execute_pick_up_plan()
        self.cram_segmenter.start()

    @staticmethod
    def execute_pick_up_plan():
        object_description = ObjectDesignatorDescription(names=["milk"])
        description = PickUpActionDescription(object_description, [Arms.LEFT], [GraspDescription(Grasp.FRONT)])
        with simulated_robot:
            plan = SequentialPlan(MoveTorsoActionDescription(TorsoState.HIGH),
            description)
            plan.perform()
        # plan.plot()
