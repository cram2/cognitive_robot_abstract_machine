from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, field
from os.path import dirname

from typing_extensions import (
    ClassVar,
    List,
    Dict,
    Any,
    TYPE_CHECKING,
    Optional,
    Callable,
    Type,
)

from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery


class ReasoningResult(UserDict[str, Any]): ...


class CaseRDRs(UserDict[Type, GeneralRDR]): ...


@dataclass
class CaseReasoner:
    """
    The case reasoner is a class that uses Ripple Down Rules to reason on case related concepts.
    The reasoner can be used in two ways:
    1. Classification mode, where the reasoner infers all concepts that it has rules for at that time.
    >>> reasoner = CaseReasoner(case)
    >>> inferred_concepts = reasoner.reason()
    >>> inferred_attribute_values = inferred_concepts['attribute_name']
    2. Fitting mode, where the reasoner prompts the expert for answers given a query on a world concept. This allows
    incremental knowledge gain, improved reasoning capabilities, and an increased breadth of application with more
     usage.
     >>> reasoner = CaseReasoner(case)
     >>> reasoner.fit_attribute("attribute_name", [attribute_types,...], False)
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
        Perform rule-based reasoning on the current semantic annotation and infer all possible concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all inferred values of that concept.
        """
        self.result = self.rdr.classify(self.case, modify_case=True)
        return self.result

    def fit_attribute(
        self,
        attribute_name: str,
        attribute_types: List[Type[Any]],
        mutually_exclusive: bool,
        update_existing_rules: bool = False,
        case_factory: Optional[Callable] = None,
        scenario: Optional[Callable] = None,
    ) -> None:
        """
        Fit the semantic annotation RDR to the required attribute types.

        :param attribute_name: The attribute name that the RDR should be fitted to.
        :param attribute_types: A list of attribute types that the RDR should be fitted to.
        :param mutually_exclusive: whether the attribute values are mutually exclusive or not.
        :param update_existing_rules: If True, existing rules of the given types will be updated with new rules,
         else they will be skipped.
        :param case_factory: Optional callable that can be used to recreate the case object.
        :param scenario: Optional callable that represents the test method or scenario that is being executed.
        """
        case_query = CaseQuery(
            self.case,
            attribute_name,
            tuple(attribute_types),
            mutually_exclusive,
            case_factory=case_factory,
            scenario=scenario,
        )
        self.rdr.fit_case(case_query, update_existing_rules=update_existing_rules)
