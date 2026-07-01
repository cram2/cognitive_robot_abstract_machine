package com.example.pyroles

import com.intellij.testFramework.fixtures.BasePlatformTestCase

/**
 * In-IDE tests for [RoleMembersProvider] against the **real shape of the krrood `Role`** —
 * the case the bundled `sample/roles_demo.py` does not cover and where the cram failure was
 * suspected.
 *
 * Unlike [RoleMembersProviderTest], which defines an inline single-file `Role(Generic[T])`,
 * each test here adds a separate `krrood/patterns/role.py` module (so an imported `Role`
 * carries the qualified name `krrood.patterns.role.Role` and the `Symbol, SubClassSafeGeneric[T]`
 * base hierarchy), then a consumer file that **imports** that `Role` across files, parameterises
 * it with `Role[Taker]`, and accesses a delegated member. Completion offering the taker's
 * members only happens if the provider resolved the taker from this krrood-style code.
 */
class KrroodRoleMembersProviderTest : BasePlatformTestCase() {

    /**
     * Adds a faithful-but-minimal `krrood.patterns.role` package. Only PSI / type resolution
     * matters to the provider, so the runtime machinery (SymbolGraph, delegation bodies) is
     * omitted; what is reproduced is the qualified name, the `Symbol, SubClassSafeGeneric[T]`
     * bases, the `__getattr__` delegation signal, and the inherited keyword-only `role_taker`
     * field that every role takes its taker through.
     */
    private fun addKrroodRoleModule() {
        myFixture.addFileToProject("krrood/__init__.py", "")
        myFixture.addFileToProject("krrood/patterns/__init__.py", "")
        myFixture.addFileToProject(
            "krrood/patterns/role.py",
            """
            from __future__ import annotations
            from dataclasses import dataclass, field
            from typing import Any, Generic, TypeVar

            T = TypeVar("T")

            class Symbol: ...

            class SubClassSafeGeneric(Generic[T]): ...

            @dataclass(eq=False)
            class Role(Symbol, SubClassSafeGeneric[T]):
                role_taker: T = field(kw_only=True)
                def __getattr__(self, item: str) -> Any: ...
            """.trimIndent(),
        )
    }

    /** `class Kitchen(Role[Room])` — the production form; the taker comes from the generic. */
    fun testConcreteGenericRoleDelegatesTakerMembers() {
        addKrroodRoleModule()
        myFixture.configureByText(
            "kitchen.py",
            """
            from __future__ import annotations
            from krrood.patterns.role import Role

            class Room:
                floor: int
                def area(self) -> float: ...

            class Kitchen(Role[Room]):
                appliances: list

            kitchen = Kitchen()
            kitchen.<caret>
            """.trimIndent(),
        )
        myFixture.completeBasic()
        val members = myFixture.lookupElementStrings ?: emptyList()
        // Delegated from Room (floor, area) and the role's own field (appliances).
        assertContainsElements(members, "floor", "area", "appliances")
    }

    /** A bare `class Kitchen(Role)` has no generic argument, so no taker can be resolved. */
    fun testBareRoleWithoutGenericArgumentDelegatesNoTakerMembers() {
        addKrroodRoleModule()
        myFixture.configureByText(
            "bare_kitchen.py",
            """
            from __future__ import annotations
            from krrood.patterns.role import Role

            class Room:
                floor: int
                def area(self) -> float: ...

            class Kitchen(Role):
                appliances: list

            kitchen = Kitchen()
            kitchen.<caret>
            """.trimIndent(),
        )
        myFixture.completeBasic()
        val members = myFixture.lookupElementStrings ?: emptyList()
        // The role's own field is still offered; the taker's members are not, since the taker
        // type is unknown without a `Role[Room]` generic argument.
        assertContainsElements(members, "appliances")
        assertDoesntContain(members, "floor", "area")
    }

    /** `class CEO(Role[TPerson])` where `TPerson = TypeVar("TPerson", bound=Person)`. */
    fun testTypeVarBoundResolvesTaker() {
        addKrroodRoleModule()
        myFixture.configureByText(
            "ceo.py",
            """
            from __future__ import annotations
            from typing import TypeVar
            from krrood.patterns.role import Role

            class Person:
                name: str
                def greet(self) -> str: ...

            TPerson = TypeVar("TPerson", bound=Person)

            class CEO(Role[TPerson]):
                perks: list

            ceo = CEO()
            ceo.<caret>
            """.trimIndent(),
        )
        myFixture.completeBasic()
        val members = myFixture.lookupElementStrings ?: emptyList()
        // The taker is the TypeVar's bound (Person), so Person's members are delegated.
        assertContainsElements(members, "name", "greet", "perks")
    }
}
