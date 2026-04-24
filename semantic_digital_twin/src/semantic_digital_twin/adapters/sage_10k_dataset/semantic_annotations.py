from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody


@dataclass(eq=False)
class Sage10kLabel(HasRootBody):
    """
    Represents a label in the Sage10k dataset annotation hierarchy.
    """
