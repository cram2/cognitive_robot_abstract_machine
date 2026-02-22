from dataclasses import dataclass, field
from os.path import dirname

from typing_extensions import Optional, List, Dict, Any, Type, Callable, ClassVar

from ..world import World
from ..world_description.world_entity import SemanticAnnotation
from .reasoner import CaseReasoner


@dataclass
class WorldReasoner:
    """
    A utility class that uses CaseReasoner for reasoning on the world concepts.
    """

    world: World
    """
    The world instance to reason on.
    """
    _last_world_model_version: Optional[int] = field(init=False, default=None)
    """
    The last world model version of the world used when :py:meth:`reason` 
    was last called.
    """
    reasoner: CaseReasoner = field(init=False)
    """
    The case reasoner that is used to reason on the world concepts.
    """
    model_directory: ClassVar[str] = dirname(__file__)
    """
    The directory where the rdr model folder is located.
    """

    def __post_init__(self):
        self.reasoner = CaseReasoner(self.world, model_directory=self.model_directory)

    def infer_semantic_annotations(self) -> List[SemanticAnnotation]:
        """
        Infer the semantic annotations of the world by calling the :py:meth:`reason` method and extracting all inferred semantic annotations.

        :return: The inferred semantic annotations of the world.
        """
        with self.world.modify_world():
            result = self.reason()
        return result.get("semantic_annotations", [])

    def reason(self) -> Dict[str, Any]:
        """
        Perform rule-based reasoning on the current world and infer all possible concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all inferred values of that concept.
        """
        if (
            self.world.get_world_model_manager().version
            != self._last_world_model_version
        ):
            self.reasoner.result = self.reasoner.rdr.classify(self.world)
            self._update_world_attributes()
            self._last_world_model_version = (
                self.world.get_world_model_manager().version
            )
        return self.reasoner.result

    def _update_world_attributes(self):
        """
        Update the world attributes from the values in the result of the latest :py:meth:`reason` call.
        """
        for attr_name, attr_value in self.reasoner.result.items():
            if isinstance(getattr(self.world, attr_name), list):
                attr_value = list(attr_value)
            if attr_name != "semantic_annotations":
                setattr(self.world, attr_name, attr_value)

    def fit_semantic_annotations(
        self,
        required_semantic_annotations: List[Type[SemanticAnnotation]],
        update_existing_semantic_annotations: bool = False,
        world_factory: Optional[Callable] = None,
        scenario: Optional[Callable] = None,
    ) -> None:
        """
        Fit the world RDR to the required semantic annotation types.

        :param required_semantic_annotations: A list of semantic annotation types that the RDR should be fitted to.
        :param update_existing_semantic_annotations: If True, existing semantic annotations will be updated with new rules, else they will be skipped.
        :param world_factory: Optional callable that can be used to recreate the world object.
        :param scenario: Optional callable that represents the test method or scenario that is being executed.
        """
        self.reasoner.fit_attribute(
            "semantic_annotations",
            required_semantic_annotations,
            False,
            update_existing_rules=update_existing_semantic_annotations,
            case_factory=world_factory,
            scenario=scenario,
        )
