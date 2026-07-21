from dataclasses import dataclass


@dataclass
class LateBoundAnnotationType:
    """
    A field type imported only under ``TYPE_CHECKING`` by another module.

    Stands in for a type (such as ``semantic_digital_twin``'s ``World``) whose defining
    module may still be partially initialised when an annotated owner first has its
    import scope cached.
    """

    name: str = ""
