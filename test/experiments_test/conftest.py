"""
The ORM build every experiment test may need.

Each experiment keeps its own tests, and the fixtures they need, in a package of its own
under this one, so that tests needing ROS and tests needing nothing but the domain live
side by side.
"""

from __future__ import annotations

# Built before anything reads a mapped datastructure: pytest imports every conftest of a
# run before calling any hook, so a hook would fire too late. The build runs once per
# process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()
