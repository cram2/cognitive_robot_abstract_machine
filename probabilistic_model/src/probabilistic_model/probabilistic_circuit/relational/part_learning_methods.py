"""
What the exchangeable parts of a relational circuit are fitted with, keyed by EQL
attribute accesses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from typing_extensions import Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    MissingPartLearningMethodError,
    NotAnAttributeAccessError,
)


@dataclass
class PartLearningMethods(SubclassJSONSerializer):
    """
    A defaultdict of learning methods keyed by attribute accesses from a variable of the
    class of a relational circuit to its exchangeable parts, such as
    ``variable(SceneBuilding).floors.rooms``.

    Every attribute access is stored under its path, such as ``"floors.rooms"``, since
    attribute accesses hash by identity and compare symbolically.
    """

    default: LearningMethod | None = field(default_factory=JointProbabilityTree)
    """
    What a part without a learning method is fitted with, copied for every such part,
    or ``None`` to require a learning method for every part.
    """

    methods_by_path: dict[str, LearningMethod] = field(init=False, default_factory=dict)
    """
    The learning method of every part, keyed by its path.
    """

    @classmethod
    def from_methods_and_default(
        cls,
        methods: dict[Attribute, LearningMethod],
        default: LearningMethod | None,
    ) -> Self:
        """
        :param methods: The learning methods of the parts.
        :param default: What a part without a learning method is fitted with, or
            ``None`` to require a learning method for every part.
        :return: The learning methods of the parts.
        """
        result = cls(default)
        for part, method in methods.items():
            result[part] = method
        return result

    @staticmethod
    def path_of(part: Attribute) -> str:
        """
        :param part: An attribute access on a variable.
        :return: The path of the attribute access.
        :raises NotAnAttributeAccessError: If the part is not an attribute access on a
            variable.
        """
        reversed_names = []
        key = part
        while isinstance(key, Attribute):
            reversed_names.append(key._attribute_name_)
            key = key._child_
        if not reversed_names or not isinstance(key, Variable):
            raise NotAnAttributeAccessError(part)
        return ".".join(reversed(reversed_names))

    def __getitem__(self, part: Attribute) -> LearningMethod:
        return self.method_of_path(self.path_of(part))

    def __setitem__(self, part: Attribute, method: LearningMethod):
        self.methods_by_path[self.path_of(part)] = method

    def __contains__(self, part: Attribute) -> bool:
        return self.path_of(part) in self.methods_by_path

    def method_of_path(self, path: str) -> LearningMethod:
        """
        :param path: The path of a part.
        :return: The learning method of the part, which is a copy of the default if the
            part has none yet.
        :raises MissingPartLearningMethodError: If the part has no learning method and
            there is no default.
        """
        if path not in self.methods_by_path:
            if self.default is None:
                raise MissingPartLearningMethodError(path)
            self.methods_by_path[path] = copy.deepcopy(self.default)
        return self.methods_by_path[path]

    def of_part(self, part: str) -> PartLearningMethods:
        """
        :param part: The field name of an exchangeable part.
        :return: The learning methods of the parts nested in the part, with the same
            default.
        """
        prefix = f"{part}."
        result = PartLearningMethods(self.default)
        result.methods_by_path = {
            path.removeprefix(prefix): method
            for path, method in self.methods_by_path.items()
            if path.startswith(prefix)
        }
        return result

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(),
            "default": to_json(self.default, **kwargs),
            "paths": list(self.methods_by_path),
            "methods": [
                to_json(method, **kwargs) for method in self.methods_by_path.values()
            ],
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        result = cls(from_json(data["default"], **kwargs))
        result.methods_by_path = {
            path: from_json(method, **kwargs)
            for path, method in zip(data["paths"], data["methods"])
        }
        return result
