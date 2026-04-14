import traceback
import uuid
from threading import Thread
from time import time
from typing import Optional

import rclpy
from py_trees import behaviour
from py_trees.behaviour import Behaviour
from line_profiler import profile
from py_trees.common import Status
from py_trees.composites import Composite
from py_trees.decorators import SuccessIsRunning

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import raise_to_blackboard


class AsyncBehavior(GiskardBehavior, Composite):
    """
    A composite that runs its children in a different thread.
    Status is Running if all children are Running.
    If one child returns either Success or Failure, this behavior will return it as well.
    """

    def __init__(self, name: str, max_hz: Optional[float] = None):
        """
        :param name:
        :param max_hz: The frequency at which this thread is looped will be limited to this value, if possible.
        """
        super().__init__(name)
        self.set_status(Status.INVALID)
        self.looped_once = False
        self.max_hz = max_hz
        if self.max_hz is not None:
            self.sleeper = rospy.node.create_rate(frequency=self.max_hz)
        else:
            self.sleeper = None

    def initialise(self) -> None:
        self.looped_once = False
        self.update_thread = Thread(
            target=self.loop_over_plugins, name=f"async {self.name}"
        )
        self.update_thread.start()
        super().initialise()

    def add_child(
        self, child: behaviour.Behaviour, success_is_running: bool = True
    ) -> uuid.UUID:
        if success_is_running:
            success_is_running_child = SuccessIsRunning(
                f"success is running\n{child.name}", child
            )
            return super().add_child(success_is_running_child)
        return super().add_child(child)

    def remove_child(self, child: behaviour.Behaviour) -> int:
        if isinstance(child.parent, SuccessIsRunning):
            return super().remove_child(child.parent)
        return super().remove_child(child)

    def is_running(self) -> bool:
        return self.status == Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        try:
            rospy.node.get_logger().info(f"avg dt was {self.sleeper.avg_dt}")
        except Exception as e:
            pass  # sometimes the sleeper is not defined yet
        self.set_status(Status.FAILURE)
        try:
            self.update_thread.join()
        except Exception as e:
            # happens when a previous plugin fails
            # logging.logwarn('terminate was called before init')
            pass
        self.stop_children()
        super().terminate(new_status)

    def stop_children(self) -> None:
        for child in self.children:
            child.stop(self.status)

    def insert_behind(
        self,
        node: Behaviour,
        left_sibling_name: Behaviour,
        success_is_running: bool = True,
    ) -> None:
        if success_is_running:
            node = SuccessIsRunning(f"success is running\n{node.name}", node)
        try:
            sibling_id = self.children.index(left_sibling_name)
        except:
            sibling_id = self.children.index(left_sibling_name.parent)
        self.insert_child(node, sibling_id + 1)

    def tick(self):
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # Required behaviour for *all* behaviours and composites is
        # for tick() to check if it isn't running and initialise
        if self.status == Status.INVALID:
            self.status = Status.RUNNING
            self.initialise()
        elif self.status != Status.RUNNING:
            # chooser specific initialisation
            # invalidate everything
            for child in self.children:
                child.stop(Status.INVALID)
            self.current_child = None
            # run subclass (user) initialisation
            # self.initialise()
        yield self

    def set_status(self, new_state: Status) -> None:
        self.status = new_state

    def tip(self):
        return GiskardBehavior.tip(self)

    @profile
    def loop_over_plugins(self) -> None:
        try:
            self.get_blackboard().runtime = time()
            while self.is_running() and rclpy.ok():
                for child in self.children:
                    if not self.is_running():
                        return
                    for node in child.tick():
                        status = node.status
                    if status is not None:
                        self.set_status(status)
                    assert (
                        self.status is not None
                    ), f"{child.name} did not return a status"
                    if not self.is_running():
                        return
                self.looped_once = True
                if self.sleeper:
                    self.sleeper.sleep()
        except Exception as e:
            traceback.print_exc()
            raise_to_blackboard(e)
