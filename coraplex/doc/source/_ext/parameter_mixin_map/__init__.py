"""
A Sphinx directive, ``parameter-mixin-map``, that shows which coraplex actions and
giskardpy motion statechart nodes take which parameter mixin.

The map is read from the code each time the documentation is built.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import pkgutil
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from string import Template
from types import ModuleType

from coraplex.robot_plans import mixins as action_mixins
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.mixins import DesignatorParameterMixin
from docutils import nodes
from giskardpy.motion_statechart import mixins as node_mixins
from krrood.patterns.path_enumeration import PathEnumeration
from krrood.utils import recursive_subclasses
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util.docutils import SphinxDirective
from typing_extensions import Self

# %% files of the extension


class ExtensionFile(PathEnumeration):
    """
    The files the extension adds to the documentation.
    """

    MARKUP = Path(__file__).parent / "parameter_mixin_map.html"
    """
    The markup the directive renders, with a ``$map`` placeholder for the map.
    """

    STATIC = Path(__file__).parent / "static"
    """
    The directory of the script and the stylesheet that draw the map in the browser.
    """


class StaticFile(StrEnum):
    """
    The files in :attr:`ExtensionFile.STATIC` that every page loads.
    """

    SCRIPT = "parameter_mixin_map.js"
    STYLESHEET = "parameter_mixin_map.css"


# %% the code the map is read from


class Framework(StrEnum):
    """
    The frameworks whose classes take parameter mixins.
    """

    CORAPLEX = "coraplex"
    GISKARDPY = "giskardpy"

    @classmethod
    def defining(cls, defined_class: type) -> Self:
        """
        :param defined_class: A class one of the frameworks defines.
        :return: The framework that defines it.
        """
        return cls(defined_class.__module__.partition(".")[0])


class ParameterTakingPackage(StrEnum):
    """
    The packages that define the actions and nodes the map shows.
    """

    ACTIONS = "coraplex.robot_plans.actions"
    TASKS = "giskardpy.motion_statechart.tasks"
    MONITORS = "giskardpy.motion_statechart.monitors"

    def import_modules(self) -> None:
        """
        Import every module of the package, so that each class it defines is a known
        subclass of the mixins it inherits.
        """
        package = importlib.import_module(self)
        for module in pkgutil.walk_packages(package.__path__, f"{self}."):
            importlib.import_module(module.name)


# %% parameters


@dataclass
class Parameter:
    """
    A parameter as a class body declares it.
    """

    name: str
    """
    The name of the dataclass field.
    """

    type_hint: str
    """
    The type hint, as written in the source.
    """

    default: str | None
    """
    The default, as written in the source, or ``None`` for a required parameter.
    """

    documentation: str
    """
    The docstring below the field, or an empty string when it has none.
    """

    @classmethod
    @cache
    def declared_by(cls, declaring_class: type) -> list[Self]:
        """
        :param declaring_class: A dataclass.
        :return: The parameters the body of *declaring_class* declares, in their order
            there.
        """
        parameter_names = {
            dataclass_field.name
            for dataclass_field in dataclasses.fields(declaring_class)
            if dataclass_field.init
        }
        source = textwrap.dedent(inspect.getsource(declaring_class))
        body = ast.parse(source).body[0].body
        return [
            cls(
                name=statement.target.id,
                type_hint=ast.unparse(statement.annotation),
                default=(
                    None if statement.value is None else ast.unparse(statement.value)
                ),
                documentation=cls.documentation_of(following),
            )
            for statement, following in zip(body, [*body[1:], None])
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in parameter_names
        ]

    @staticmethod
    def documentation_of(statement: ast.stmt | None) -> str:
        """
        :param statement: The statement that follows a field declaration, or ``None``
            when the field ends the class body.
        :return: The text of *statement* on one line when it is a docstring, otherwise an
            empty string.
        """
        if (
            not isinstance(statement, ast.Expr)
            or not isinstance(statement.value, ast.Constant)
            or not isinstance(statement.value.value, str)
        ):
            return ""
        return " ".join(statement.value.value.split())


# %% parameter mixins


@dataclass
class ParameterMixin:
    """
    A mixin that declares parameters of actions or nodes.
    """

    name: str
    """
    The name of the mixin class.
    """

    framework: Framework
    """
    The framework that declares the mixin.
    """

    section: str
    """
    The heading of the section of its module that declares the mixin.
    """

    documentation: str
    """
    The docstring of the mixin, on one line.
    """

    parameters: list[Parameter]
    """
    The parameters the mixin declares.
    """


@dataclass
class MixinModule(ABC):
    """
    A module that declares parameter mixins.
    """

    module: ModuleType
    """
    The module.
    """

    @abstractmethod
    def is_parameter_mixin(self, candidate: type) -> bool:
        """
        :param candidate: A class the module defines.
        :return: Whether *candidate* is a parameter mixin.
        """

    def defined_classes(self) -> list[type]:
        """
        :return: The classes the module defines, in their order there.
        """
        return [
            candidate
            for candidate in vars(self.module).values()
            if inspect.isclass(candidate)
            and candidate.__module__ == self.module.__name__
        ]

    def parameter_mixin_classes(self) -> list[type]:
        """
        :return: The parameter mixins the module defines, in their order there.
        """
        return [
            candidate
            for candidate in self.defined_classes()
            if self.is_parameter_mixin(candidate)
        ]

    def parameter_mixins(self) -> list[ParameterMixin]:
        """
        :return: The parameter mixins the module defines, in their order there.
        """
        return [
            ParameterMixin(
                name=mixin.__name__,
                framework=Framework.defining(mixin),
                section=self.section_of(mixin),
                documentation=" ".join(inspect.getdoc(mixin).split()),
                parameters=Parameter.declared_by(mixin),
            )
            for mixin in self.parameter_mixin_classes()
        ]

    def section_of(self, mixin: type) -> str:
        """
        :param mixin: A class the module defines.
        :return: The heading of the ``# %%`` section that declares *mixin*.
        """
        declaration_line = inspect.getsourcelines(mixin)[1]
        preceding_lines = inspect.getsource(self.module).splitlines()[:declaration_line]
        return [
            line.removeprefix("# %%").strip()
            for line in preceding_lines
            if line.startswith("# %%")
        ][-1]


@dataclass
class ActionMixinModule(MixinModule):
    """
    The module of the coraplex action parameter mixins, which also defines the
    behaviours that actions share.
    """

    def is_parameter_mixin(self, candidate: type) -> bool:
        return (
            issubclass(candidate, DesignatorParameterMixin)
            and candidate is not DesignatorParameterMixin
            and not issubclass(candidate, Action)
        )

    def behaviours(self) -> list[type]:
        """
        :return: The behaviours, partial actions that whole actions inherit, the module
            defines.
        """
        return [
            candidate
            for candidate in self.defined_classes()
            if issubclass(candidate, Action)
        ]


@dataclass
class NodeMixinModule(MixinModule):
    """
    The module of the giskardpy motion statechart node parameter mixins.
    """

    def is_parameter_mixin(self, candidate: type) -> bool:
        return True


# %% classes taking parameter mixins


@dataclass
class RedeclaredParameter:
    """
    A parameter of a mixin that a class between the mixin and a class taking it declares
    again, for example with another default.
    """

    declaring_class: str
    """
    The name of the class that declares the parameter again.
    """

    parameter: Parameter
    """
    The parameter as that class declares it.
    """

    @classmethod
    def nearest_to(cls, taking_class: type, parameter_name: str) -> Self:
        """
        :param taking_class: A class that takes the parameter.
        :param parameter_name: The name of the parameter.
        :return: The declaration of the parameter nearest to *taking_class* in its
            method resolution order.
        """
        return next(
            cls(declaring_class=ancestor.__name__, parameter=parameter)
            for ancestor in taking_class.__mro__
            if dataclasses.is_dataclass(ancestor)
            for parameter in Parameter.declared_by(ancestor)
            if parameter.name == parameter_name
        )


@dataclass
class ParameterisedClass:
    """
    An action or a motion statechart node, with the parameter mixins it takes.
    """

    name: str
    """
    The name of the class.
    """

    framework: Framework
    """
    The framework that defines the class.
    """

    defining_module: str
    """
    The last component of the name of the module that defines the class.
    """

    parameter_mixins: list[str]
    """
    The names of the parameter mixins the class takes.
    """

    behaviours: list[str]
    """
    The names of the behaviours the class inherits.
    """

    redeclared_parameters: list[RedeclaredParameter]
    """
    The parameters of its mixins that are declared again on the way to the class.
    """

    @classmethod
    def from_class(
        cls,
        taking_class: type,
        parameter_mixins: list[type],
        behaviours: list[type],
    ) -> Self:
        """
        :param taking_class: An action or node.
        :param parameter_mixins: Every parameter mixin.
        :param behaviours: Every behaviour.
        :return: *taking_class* with the parameter mixins and behaviours it inherits.
        """
        taken_mixins = [
            mixin for mixin in parameter_mixins if issubclass(taking_class, mixin)
        ]
        return cls(
            name=taking_class.__name__,
            framework=Framework.defining(taking_class),
            defining_module=taking_class.__module__.rpartition(".")[2],
            parameter_mixins=[mixin.__name__ for mixin in taken_mixins],
            behaviours=[
                behaviour.__name__
                for behaviour in behaviours
                if issubclass(taking_class, behaviour)
            ],
            redeclared_parameters=[
                declaration
                for mixin in taken_mixins
                for parameter in Parameter.declared_by(mixin)
                if (
                    declaration := RedeclaredParameter.nearest_to(
                        taking_class, parameter.name
                    )
                ).declaring_class
                != mixin.__name__
            ],
        )


# %% the map


@dataclass
class ParameterMixinMap:
    """
    Which actions and nodes take which parameter mixin.
    """

    parameter_mixins: list[ParameterMixin]
    """
    The coraplex parameter mixins, followed by the giskardpy ones.
    """

    classes: list[ParameterisedClass]
    """
    The concrete actions and nodes that take at least one parameter mixin.
    """

    @classmethod
    def from_code(cls) -> Self:
        """
        :return: The map of the code that is installed.
        """
        for package in ParameterTakingPackage:
            package.import_modules()
        action_module = ActionMixinModule(action_mixins)
        modules = [action_module, NodeMixinModule(node_mixins)]
        mixin_classes = [
            mixin for module in modules for mixin in module.parameter_mixin_classes()
        ]
        taking_classes = {
            subclass
            for mixin in mixin_classes
            for subclass in recursive_subclasses(mixin)
            if cls.is_shown(subclass)
        }
        return cls(
            parameter_mixins=[
                mixin for module in modules for mixin in module.parameter_mixins()
            ],
            classes=[
                ParameterisedClass.from_class(
                    taking_class, mixin_classes, action_module.behaviours()
                )
                for taking_class in sorted(
                    taking_classes, key=lambda taking_class: taking_class.__name__
                )
            ],
        )

    @staticmethod
    def is_shown(candidate: type) -> bool:
        """
        :param candidate: A class that inherits a parameter mixin.
        :return: Whether *candidate* is a concrete action or node of one of the
            :class:`ParameterTakingPackage`.
        """
        return (
            candidate.__module__.startswith(tuple(ParameterTakingPackage))
            and not inspect.isabstract(candidate)
            and ABC not in candidate.__bases__
            and not candidate.__name__.startswith("_")
        )

    def to_json(self) -> str:
        """
        :return: The map as JSON that can be embedded in a ``script`` element.
        """
        return json.dumps(dataclasses.asdict(self)).replace("<", "\\u003c")


# %% directive


class ParameterMixinMapDirective(SphinxDirective):
    """
    Renders the :class:`ParameterMixinMap` of the code the documentation is built from.
    """

    def run(self) -> list[nodes.Node]:
        self.env.note_reread()
        markup = Template(ExtensionFile.MARKUP.read_text()).substitute(
            map=ParameterMixinMap.from_code().to_json()
        )
        return [nodes.raw("", markup, format="html")]


def add_static_directory(app: Sphinx, config: Config) -> None:
    """
    Serve the script and the stylesheet of the extension with the documentation.

    :param app: The Sphinx application.
    :param config: The configuration of the build.
    """
    config.html_static_path.append(str(ExtensionFile.STATIC))


def setup(app: Sphinx) -> dict[str, bool]:
    """
    Register the directive and the files it needs.

    :param app: The Sphinx application.
    :return: The parallel safety of the extension.
    """
    app.add_directive("parameter-mixin-map", ParameterMixinMapDirective)
    app.connect("config-inited", add_static_directory)
    app.add_js_file(StaticFile.SCRIPT, defer="defer")
    app.add_css_file(StaticFile.STYLESHEET)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
