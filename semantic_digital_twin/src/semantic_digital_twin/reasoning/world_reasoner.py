from dataclasses import dataclass, field
from os.path import dirname

from typing_extensions import Optional, List, Dict, Any, ClassVar

from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.reasoning.reasoner import CaseReasoner


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
    The last world model version of the world used when :py:meth:`reason` was last
    called.
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
        Infer the semantic annotations of the world by calling the :py:meth:`reason`
        method and extracting all inferred semantic annotations.

        :return: The inferred semantic annotations of the world.
        """
        with self.world.modify_world():
            result = self.reason()
        return result.get("semantic_annotations", [])

    def reason(self) -> Dict[str, Any]:
        """
        Perform rule-based reasoning on the current world and infer all possible
        concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all
            inferred values of that concept.
        """
        if (
            self.world.get_world_model_manager().version
            != self._last_world_model_version
        ):
            self.reasoner.result = self.reasoner.rdr.classify(self.world)
            with self.world.modify_world():
                self._update_world_attributes()
            self._last_world_model_version = (
                self.world.get_world_model_manager().version
            )
        return self.reasoner.result

    def _update_world_attributes(self):
        """
        Update the world attributes from the values in the result of the latest
        :py:meth:`reason` call.
        """
        for attr_name, attr_value in self.reasoner.result.items():
            if isinstance(getattr(self.world, attr_name), list):
                attr_value = list(attr_value)
            if attr_name != "semantic_annotations":
                setattr(self.world, attr_name, attr_value)
            else:
                for semantic_annotation in attr_value:
                    self._hold_in_world(semantic_annotation)

    def _hold_in_world(self, semantic_annotation: SemanticAnnotation) -> None:
        """
        Give the world the inferred annotation, and the joint that already moves it.

        An inferred annotation the world already holds an equal of is dropped in favour
        of the one the world holds, so that reasoning over a world that was annotated
        before does not store a second copy of what it recognises, and so that the joint
        is given to the annotation everything else refers to.
        """
        annotation_in_world = self.world.get_semantic_annotation_equal_to(
            semantic_annotation
        )
        if annotation_in_world is None:
            self.world.add_semantic_annotation_recursively(semantic_annotation)
            annotation_in_world = semantic_annotation
        if isinstance(annotation_in_world, HasMechanicalJoint):
            annotation_in_world.create_default_mechanical_joint()
