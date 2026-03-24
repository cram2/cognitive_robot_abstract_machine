from dataclasses import field, dataclass

@dataclass(eq=False)
class RoleTakerInAnotherModuleRoleAttributes:
    introduced_attribute: str = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModuleMixin(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModule(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str
