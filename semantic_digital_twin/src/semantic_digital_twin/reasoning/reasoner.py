from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, field
from os.path import dirname

from typing_extensions import (
    ClassVar,
    Dict,
    Any,
    Optional,
    Type,
)

from krrood.ripple_down_rules.rdr import GeneralRDR


class ReasoningResult(UserDict[str, Any]): ...


class CaseRDRs(UserDict[Type, GeneralRDR]): ...


@dataclass
class CaseReasoner:
    """
    Uses Ripple Down Rules to infer concepts about a case.

    The rules are read from the model directory and applied to the case, inferring every
    concept they have a rule for:

    >>> reasoner = CaseReasoner(case)
    >>> inferred_concepts = reasoner.reason()
    >>> inferred_attribute_values = inferred_concepts['attribute_name']
    """

    case: Any
    """
    The case instance on which the reasoning is performed.
    """

    result: Optional[ReasoningResult] = field(init=False, default=None)
    """
    The latest result of the :py:meth:`reason` call.
    """

    model_directory: str = field(default_factory=lambda: dirname(__file__))
    """
    The directory where the rdr model folder is located.
    """

    rdrs: ClassVar[CaseRDRs] = CaseRDRs()
    """
    This is a collection of ripple down rules reasoners that infer case attributes.
    """

    def __post_init__(self):
        if self.case.__class__ not in self.rdrs:
            self.rdrs[self.case.__class__] = GeneralRDR(
                save_dir=self.model_directory,
                model_name=f"{self.case.__class__.__name__.lower()}_rdr",
            )

    @property
    def rdr(self) -> GeneralRDR:
        """
        The Ripple Down Rules instance that is used for reasoning on the case concepts.

        :return: The Ripple Down Rules instance.
        """
        return self.rdrs[self.case.__class__]

    def reason(self) -> Dict[str, Any]:
        """
        Perform rule-based reasoning on the current semantic annotation and infer all
        possible concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all
            inferred values of that concept.
        """
        self.result = self.rdr.classify(self.case, modify_case=True)
        return self.result
