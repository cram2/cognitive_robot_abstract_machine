from __future__ import annotations

from enum import auto, Enum

from typing_extensions import List, Dict, Any, Type

from krrood.ripple_down_rules.utils import SubclassJSONSerializer


class InferMode(Enum):
    """
    The infer mode of a predicate, whether to infer new relations or retrieve current relations.
    """
    Auto = auto()
    """
    Inference is done automatically depending on the world state.
    """
    Always = auto()
    """
    Inference is always performed.
    """
    Never = auto()
    """
    Inference is never performed.
    """


class ExitStatus(Enum):
    """
    Describes the status at exit of the user interface.
    """
    CLOSE = auto()
    """
    The user wants to stop the program.
    """
    SUCCESS = auto()
    """
    The user completed the task successfully.
    """

class InteractionMode(Enum):
    """
    The interaction mode of the RDR.
    """
    IPythonOnly = auto()
    """
    IPythonOnly mode, the mode where the user uses only an Ipython shell to interact with the RDR.
    """
    GUI = auto()
    """
    GUI mode, the mode where the user uses a GUI to interact with the RDR.
    """


class Editor(str, Enum):
    """
    The editor that is used to edit the rules.
    """
    Pycharm = "pycharm"
    """
    PyCharm editor.
    """
    Code = "code"
    """
    Visual Studio Code editor.
    """
    CodeServer = "code-server"
    """
    Visual Studio Code server editor.
    """
    @classmethod
    def from_str(cls, editor: str) -> Editor:
        """
        Convert a string value to an Editor enum.

        :param editor: The string that represents the editor name.
        :return: The Editor enum.
        """
        if editor not in cls._value2member_map_:
            raise ValueError(f"Editor {editor} is not supported.")
        return cls._value2member_map_[editor]


class Category(str, SubclassJSONSerializer, Enum):

    @classmethod
    def from_str(cls, value: str) -> Category:
        return getattr(cls, value)

    @classmethod
    def from_strs(cls, values: List[str]) -> List[Category]:
        return [cls.from_str(value) for value in values]

    @property
    def as_dict(self):
        return {self.__class__.__name__.lower(): self.value}

    def _to_json(self) -> Dict[str, Any]:
        return self.as_dict

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Category:
        return cls.from_str(data[cls.__name__.lower()])


class Stop(Category):
    """
    A stop category is a special category that represents the stopping of the classification to prevent a wrong
    conclusion from being made.
    """
    stop = "stop"


class PromptFor(Enum):
    """
    The reason of the prompt. (e.g. get conditions, conclusions, or affirmation).
    """
    Conditions: str = "conditions"
    """
    Prompt for rule conditions about a case.
    """
    Conclusion: str = "value"
    """
    Prompt for rule conclusion about a case.
    """
    Affirmation: str = "affirmation"
    """
    Prompt for rule conclusion affirmation about a case.
    """

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class MCRDRMode(Enum):
    """
    The modes of the MultiClassRDR.
    """
    StopOnly = auto()
    """
    StopOnly mode, stop wrong conclusion from being made and does not add a new rule to make the correct conclusion.
    """
    StopPlusRule = auto()
    """
    StopPlusRule mode, stop wrong conclusion from being made and adds a new rule with same conditions as stopping rule
     to make the correct conclusion.
    """
    StopPlusRuleCombined = auto()
    """
    StopPlusRuleCombined mode, stop wrong conclusion from being made and adds a new rule with combined conditions of
    stopping rule and the rule that should have fired.
    """


class RDREdge(Enum):
    Refinement = "except if"
    """
    Refinement edge, the edge that represents the refinement of an incorrectly fired rule.
    """
    Alternative = "else if"
    """
    Alternative edge, the edge that represents the alternative to the rule that has not fired.
    """
    Next = "next"
    """
    Next edge, the edge that represents the next rule to be evaluated.
    """
    Filter = "filter if"
    """
    Filter edge, the edge that represents the filter condition.
    """
    Empty = ""
    """
    Empty edge, used for example for the root/input node of the tree.
    """

    @classmethod
    def from_value(cls, value: str) -> RDREdge:
        """
        Convert a string value to an RDREdge enum.

        :param value: The string that represents the edge type.
        :return: The RDREdge enum.
        """
        if value not in cls._value2member_map_:
            raise ValueError(f"RDREdge {value} is not supported.")
        return cls._value2member_map_[value]
