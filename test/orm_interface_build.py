"""
Building the ORM interfaces a test run needs.

Every package's ``ormatic_interface.py`` is generated rather than tracked, so a checkout
holds none until something builds them. Only the packages whose tests read a mapped
datastructure call this; the rest never pay for a build they would not read.

A build takes about a minute, so a run only pays for it when the checkout has not built
its interfaces since the sources changed. ``--orm-build`` overrides that.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from functools import cache

from typing_extensions import Optional, Sequence, Tuple

from cognitive_robot_abstract_machine.exceptions import (
    MissingOrmBuildChoiceError,
    UnknownOrmBuildChoiceError,
)
from cognitive_robot_abstract_machine.orm_interfaces import (
    WORKSPACE_ORM_INTERFACES,
    WorkspaceOrmInterfaces,
)

from .pytest_environment import PytestEnvironmentVariable

ORM_BUILD_OPTION = "--orm-build"
"""
The command line option a run states its choice with.
"""

OPTION_MARKER = "--"
"""
What a long option begins with, and the shortest an abbreviation of one can get.
"""

CHOICE_SEPARATOR = "="
"""
What joins a choice to the option on a command line.
"""


def names_the_orm_build_option(argument: str) -> bool:
    """
    Whether an argument names the option, spelled in full or cut short the way argparse
    lets a run abbreviate a long option.

    :param argument: The argument, with any choice joined to it already taken off.
    """
    if len(argument) <= len(OPTION_MARKER):
        return False
    return ORM_BUILD_OPTION.startswith(argument)


class OrmBuild(StrEnum):
    """
    When a run builds the ORM interfaces it reads.
    """

    AUTO = "auto"
    """
    Builds only what the checkout has not built since its sources changed.
    """

    ALWAYS = "always"
    """
    Builds every run, whatever the checkout holds.
    """

    NEVER = "never"
    """
    Builds nothing, and reads whatever the checkout holds.
    """

    @classmethod
    def choices(cls) -> Tuple[str, ...]:
        """
        The choices a run can state, in the order the help of the option lists them.
        """
        return tuple(choice.value for choice in cls)

    @classmethod
    def named(cls, choice: str, source: str) -> OrmBuild:
        """
        Read the choice a value names.

        :param choice: The value to read.
        :param source: Where it was read, for a rejection to name.
        :return: The choice it names.
        :raises UnknownOrmBuildChoiceError: If it names none.
        """
        if choice not in cls.choices():
            raise UnknownOrmBuildChoiceError(source, choice, cls.choices())
        return cls(choice)

    @classmethod
    def stated_on(cls, arguments: Sequence[str]) -> Optional[OrmBuild]:
        """
        Read the choice a command line states, in either of the spellings argparse
        accepts.

        :param arguments: The arguments the run was started with.
        :return: The choice they state, or nothing if they state none.
        :raises MissingOrmBuildChoiceError: If they name the option and stop there.
        :raises UnknownOrmBuildChoiceError: If they state a choice that does not exist.
        """
        for position, argument in enumerate(arguments):
            option, separator, choice = argument.partition(CHOICE_SEPARATOR)
            if not names_the_orm_build_option(option):
                continue
            if separator:
                return cls.named(choice, option)
            if position + 1 == len(arguments):
                raise MissingOrmBuildChoiceError(option, cls.choices())
            return cls.named(arguments[position + 1], option)
        return None

    @classmethod
    def requested(cls) -> OrmBuild:
        """
        Read the choice this run was started with, taking the command line over the
        environment.

        ..note:: The conftests build while pytest is still importing them, before it has
            parsed the command line, so the arguments are read as they were given.

        :raises MissingOrmBuildChoiceError: If the command line names the option and
            stops there.
        :raises UnknownOrmBuildChoiceError: If either states a choice that does not
            exist.
        """
        stated = cls.stated_on(sys.argv)
        if stated is not None:
            return stated
        stated_in_environment = os.environ.get(PytestEnvironmentVariable.ORM_BUILD)
        if stated_in_environment is None:
            return cls.AUTO
        return cls.named(stated_in_environment, PytestEnvironmentVariable.ORM_BUILD)

    def builds(self, interfaces: WorkspaceOrmInterfaces) -> bool:
        """
        Whether this choice has the given interfaces built.

        :param interfaces: The interfaces of the checkout under test.
        """
        if self is OrmBuild.NEVER:
            return False
        if self is OrmBuild.ALWAYS:
            return True
        return interfaces.is_outdated


@cache
def regenerate_orm_interfaces() -> bool:
    """
    Build the ORM interfaces of this checkout, as far as the run asked for, once per
    process and never on a worker.

    The controller imports the conftests before it starts any worker, so building there
    leaves the interfaces on disk by the time a worker imports a mapped datastructure.
    Letting the workers build too would set several processes writing the same files at
    once, each paying for the whole build again. A run covering several packages asks
    once per package, and every generator runs in the first of those calls.

    :return: Whether this call built them.
    """
    if os.environ.get(PytestEnvironmentVariable.XDIST_WORKER):
        return False
    if not OrmBuild.requested().builds(WORKSPACE_ORM_INTERFACES):
        return False
    WORKSPACE_ORM_INTERFACES.regenerate()
    return True
