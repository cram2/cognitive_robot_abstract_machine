import threading
import time
from datetime import timedelta

from pycram.robot_plans import PickUpActionDescription, MoveTorsoActionDescription
from semantic_digital_twin.datastructures.joint_state import TorsoState
from semantic_digital_twin.world import World
from typing_extensions import Optional

from pycram.datastructures.enums import Arms, Grasp
from pycram.datastructures.grasp import GraspDescription
from pycram.designator import ObjectDesignatorDescription

from pycram.language import SequentialPlan

from pycram.plan import Plan
#from pycram.process_module import ProcessModule, simulated_robot


from ..episode_player import EpisodePlayer


class CRAMPlayer(EpisodePlayer):
    world: World

    def __init__(self, world: Optional[World] = None,
                 time_between_frames: timedelta = timedelta(seconds=0.5)):
        """
        Initialize the CRAM player.

        :param world: The pycram world to use.
        :param time_between_frames: The time between frames.
        """
        super().__init__(time_between_frames)
        self._init_world(world)
        self.kill_event = threading.Event()
        ProcessModule.execution_delay = time_between_frames

    def _init_world(self, world: Optional[World] = None):
        """
        Initialize the world for the CRAM player.
        """
        self.world = world if world else World.current_world
        if not self.world:
            self.world = World()
    
    def _pause(self):
        print("Pausing Plan")


    def _resume(self):
        print("Resuming Plan")

    def _run(self):
        self.ready = True
        object_description = ObjectDesignatorDescription(names=["milk"])
        description = PickUpActionDescription(object_description, [Arms.LEFT], [GraspDescription(Grasp.FRONT)])
        with simulated_robot:
            plan = SequentialPlan(MoveTorsoActionDescription(TorsoState.HIGH),
                                  description)
            plan.perform()
        plan.plot()
        # while self.world.current_world is not None:
        #     if self.kill_event.is_set():
        #         break
        #     time.sleep(1)
