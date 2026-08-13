"""Behavior tree visualization using DOT format.

This module provides functionality to render behavior trees in DOT format for visualization.
It supports both one-time rendering and decorator-based automatic rendering on status changes.

The module provides:

* Directory management for output
* Threaded rendering for performance
* Customizable rendering triggers
* Tree traversal utilities
"""

import os
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer

from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.decorators import Decorator
from typing_extensions import Optional, List, Union

from robokudo.display import render_dot_tree
from robokudo.utils.tree import find_root


def create_dir_if_not_exists(path: str) -> None:
    """Create directory if it doesn't exist.

    :param path: Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def render_now(behaviour: Union["RenderTreeToDot", "RenderTreeToDotDecorator"]) -> None:
    """Generate behavior tree snapshot and save to disk.

    This method:
    * Creates output directory if needed
    * Finds root of tree
    * Renders tree to DOT format
    * Updates rendering statistics

    :param behaviour: Behavior node requesting the render
    """
    start_timer = default_timer()

    if behaviour.create_dir_for_path:
        create_dir_if_not_exists(behaviour.path)

    # Go up until we find the root
    root = find_root(behaviour)

    render_dot_tree(
        root,
        name=f"RKTree{behaviour.suffix}-{behaviour.counter}",
        threadpool_executor=behaviour.executor,
        path_prefix=behaviour.path,
    )

    behaviour.counter += 1
    end_timer = default_timer()
    behaviour.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"


class RenderTreeToDot(Behaviour):
    """Behavior that renders tree to DOT format when ticked.

    This behavior renders the entire tree to DOT format each time it is ticked,
    saving the output to the specified directory.
    """

    def __init__(self, path: Optional[str] = None, suffix: str = "") -> None:
        """Initialize render behavior.

        :param path: Output directory path
        :param suffix: Suffix for output filenames
        """
        super().__init__(name="RenderToDot")

        self.path: Optional[str] = path
        """Output directory path"""

        self.counter: int = 0
        """Number of renders performed"""

        self.create_dir_for_path: bool = True
        """Whether to create output directory"""

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
        """Thread pool for rendering"""

        self.suffix: str = suffix
        """Suffix to append to output filenames"""

    def update(self) -> Status:
        """Render tree on each tick.

        :return: Always returns SUCCESS
        """
        render_now(self)
        return Status.SUCCESS


class RenderTreeToDotDecorator(Decorator):
    """Decorator that renders tree when child returns specific status.

    This decorator monitors its child's status and triggers a tree render
    when the status matches configured triggers.
    """

    def __init__(
        self,
        child: Optional[Behaviour] = None,
        path: Optional[str] = None,
        suffix: str = "",
        trigger_when_status_is: Optional[List[Status]] = None,
    ) -> None:
        """Initialize render decorator.

        :param child: Child behavior to monitor
        :param path: Output directory path
        :param suffix: Suffix for output filenames
        :param trigger_when_status_is: Status values that trigger rendering
        """
        super().__init__(name="RenderToDotDecorator", child=child)

        if trigger_when_status_is is None:
            trigger_when_status_is = [
                Status.SUCCESS,
                Status.FAILURE,
            ]

        self.trigger_when_status_is: List[Status] = trigger_when_status_is
        """List of status values that trigger rendering"""

        self.path: str = path
        """Output directory path"""

        self.counter: int = 0
        """Number of renders performed"""

        self.create_dir_for_path: bool = True
        """Whether to create output directory"""

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
        """Thread pool for rendering"""

        self.suffix: str = suffix
        """Suffix to append to output filenames"""

    def update(self) -> Status:
        """Check child status and render if triggered.

        :return: Status of decorated child
        """
        if self.decorated.status in self.trigger_when_status_is:
            render_now(self)

        return self.decorated.status
