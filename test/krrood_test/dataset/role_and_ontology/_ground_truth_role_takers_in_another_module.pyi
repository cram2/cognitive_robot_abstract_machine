from dataclasses import dataclass, field

@dataclass(eq=False)
class RoleTakerInAnotherModuleRoleAttributes:
    introduced_attribute: str = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModuleMixin(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModule(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str
