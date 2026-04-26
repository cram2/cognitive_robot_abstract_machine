"""Enhanced parallel behavior tree components.

This module provides enhanced parallel behavior tree components that improve upon
the standard py_trees parallel implementation. It supports:

* Configurable success policies
* Synchronization options
* Child status management
* Improved interrupt handling

The module is primarily used for:

* Complex parallel task execution
* Synchronized behavior coordination
* Robust failure handling
"""

# thanks to simon
# https://github.com/SemRoCo/giskardpy/blob/devel/src/giskardpy/tree/composites/better_parallel.py

from py_trees.composites import Parallel as OriginalParallel
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from typing_extensions import List, Union, Any, Iterator


class ParallelPolicy(object):
    """Configurable policies for parallel behavior execution.

    This class provides policy configurations that determine how parallel
    behaviors complete based on their children's status.
    """

    class Base(object):
        """Base class for parallel policies.

        .. warning::
           Should never be used directly. Use derived policy classes instead.
        """

        def __init__(self, synchronise: bool = False) -> None:
            """Initialize base policy."""

            self.synchronise: bool = synchronise
            """Stop ticking successful children until policy met"""

    class SuccessOnAll(Base):
        """Policy requiring all children to succeed.

        Returns SUCCESS only when each child returns SUCCESS. With synchronization,
        successful children are skipped until all succeed or one fails.
        """

        def __init__(self, synchronise: bool = True) -> None:
            """Initialize SuccessOnAll policy.

            :param synchronise: Stop ticking successful children until all succeed
            """
            super().__init__(synchronise=synchronise)

    class SuccessOnOne(Base):
        """Policy requiring only one child to succeed.

        Returns SUCCESS when at least one child succeeds and others are RUNNING.
        """

        def __init__(self) -> None:
            """Initialize SuccessOnOne policy.

            No configuration needed as synchronization is always disabled.
            """
            super().__init__(synchronise=False)

    class SuccessOnSelected(Base):
        """Policy requiring specific children to succeed.

        Returns SUCCESS when all specified children succeed. With synchronization,
        successful children are skipped until all specified succeed or one fails.
        """

        def __init__(self, children: List[Behaviour], synchronise: bool = True) -> None:
            """Initialize SuccessOnSelected policy.

            :param children: List of children that must succeed
            :param synchronise: Stop ticking successful children until specified succeed
            """
            super().__init__(synchronise=synchronise)
            self.children = children


class Parallel(OriginalParallel):
    """Enhanced parallel behavior tree node.

    This class extends py_trees.composites.Parallel with:

    * Improved policy handling
    * Better child status management
    * Proper interrupt propagation

    .. note::
       Children are ticked in sequence but may run concurrently.
    """

    def tick(self) -> Iterator[Union[OriginalParallel, Any]]:
        """Tick all children according to policy.

        This method:
        * Initializes if not running
        * Ticks each child according to policy
        * Updates status based on children and policy
        * Handles interrupts for running children

        :yield: Reference to self or children during traversal
        """
        if self.status != Status.RUNNING:
            # subclass (user) handling
            self.initialise()
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # process them all first
        for child in self.children:
            if self.policy.synchronise and child.status == Status.SUCCESS:
                continue
            for node in child.tick():
                yield node
        # new_status = Status.SUCCESS if self.policy == common.ParallelPolicy.SUCCESS_ON_ALL else Status.RUNNING
        new_status = Status.RUNNING
        if any([c.status == Status.FAILURE for c in self.children]):
            new_status = Status.FAILURE
        else:
            if isinstance(self.policy, ParallelPolicy.SuccessOnAll):
                if all([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
            elif isinstance(self.policy, ParallelPolicy.SuccessOnOne):
                if any([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
        # special case composite - this parallel may have children that are still running
        # so if the parallel itself has reached a final status, then these running children
        # need to be made aware of it too
        if new_status != Status.RUNNING:
            for child in self.children:
                if child.status == Status.RUNNING:
                    # interrupt it (exactly as if it was interrupted by a higher priority)
                    child.stop(Status.INVALID)
            self.stop(new_status)
        self.status = new_status
        yield self
