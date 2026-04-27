from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody


@dataclass(eq=False)
class NaturalLanguageDescription(HasRootBody):
    """
    Annotation for descriptions in natural language, e. g.
    `WordNet <https://www.nltk.org/howto/wordnet.html>`_ concepts.
    """

    description: str = field(kw_only=True)
    """
    The natural language description of root entity.
    """
