"""Object knowledge base for IAI kitchen environment.

This module provides object knowledge definitions for the IAI kitchen environment,
specifically focusing on common kitchen objects and their features. It defines
object dimensions, poses, and part relationships.

The knowledge base includes:

* Object definitions with dimensions
* Part relationships and hierarchies
* Feature specifications
* Pose information for objects and parts

.. note::
    This knowledge base is specifically designed for the IAI kitchen environment
    and includes detailed specifications for objects like mugs with their
    openings and bottoms.
"""

from dataclasses import dataclass

from robokudo.object_knowledge_base import BaseObjectKnowledgeBase, ObjectKnowledge


@dataclass
class ObjectKnowledgeBase(BaseObjectKnowledgeBase):
    """Object knowledge base for IAI kitchen environment.

    This class defines a knowledge base containing object specifications for
    the IAI kitchen environment. It includes detailed object definitions with
    their physical properties and part relationships.

    The knowledge base includes:

    * Mug object with opening and bottom parts
    * Precise dimensions for all objects
    * Pose information for object parts
    * Part relationship definitions

    .. note::
        The dimensions and poses are calibrated for the IAI kitchen setup
        and may need adjustment for other environments.
    """

    def __init__(self) -> None:
        """Initialize the IAI kitchen object knowledge base.

        Creates object definitions for:

        * Mug opening (top part)
        * Mug bottom (base part)
        * Complete mug with parts
        """
        super().__init__()

        mug_opening = ObjectKnowledge(
            name="MugOpening",
            position_x=0.0,
            position_y=0.0,
            position_z=0.08,
            orientation_x=0,
            orientation_y=0,
            orientation_z=0,
            orientation_w=1,
            x_size=0.1,
            y_size=0.1,
            z_size=0.02,
        )
        mug_bottom = ObjectKnowledge(
            name="MugBottom",
            position_x=0.0,
            position_y=0.0,
            position_z=-0.08,
            orientation_x=0,
            orientation_y=0,
            orientation_z=0,
            orientation_w=1,
            x_size=0.1,
            y_size=0.1,
            z_size=0.02,
        )
        mug = ObjectKnowledge(
            name="Mug",
            x_size=0.1,
            y_size=0.1,
            z_size=0.18,
            features=[mug_opening, mug_bottom],
        )

        self.add_entries([mug_opening, mug, mug_bottom])
