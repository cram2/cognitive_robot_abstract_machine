from py_trees.composites import Sequence

from giskardpy.tree.behaviors.cleanup import CleanUp
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.set_move_result import SetMoveResult


class PostProcessing(Sequence):
    def __init__(self, name: str = "post processing"):
        super().__init__(name, memory=True)
        self.add_child(SetMoveResult("set move result", "Planning"))
        self.add_child(ClearBlackboardException("clear exception"))
        self.add_child(CleanUp("post goal cleanup"))
