from dataclasses import dataclass, field

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role import Role
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class RoleForPose(Role[Pose], Pose):

    pose: Pose = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Pose] | Pose:
        return variable_from(cls).pose
