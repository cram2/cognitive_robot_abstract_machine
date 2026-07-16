package com.example.pyroles

import com.intellij.testFramework.fixtures.BasePlatformTestCase

/**
 * Light in-IDE tests for [RoleMembersProvider] using the Python plugin's PSI.
 *
 * Each test configures a self-contained Python file with a `Role[...]` subclass, places the
 * caret after a member access, and asserts that the role taker's members are offered by the
 * platform's completion — which only happens if the provider injected them.
 */
class RoleMembersProviderTest : BasePlatformTestCase() {

    fun testRoleDelegatesTakerMembersToCompletion() {
        myFixture.configureByText(
            "roles.py",
            """
            from __future__ import annotations
            from typing import Any, Generic, TypeVar

            T = TypeVar("T")

            class Role(Generic[T]):
                def __getattr__(self, name: str) -> Any: ...

            class Entity:
                serial: int

            class Person(Entity):
                name: str
                def greet(self) -> str: ...

            class Teacher(Role[Person]):
                role_taker: Person
                courses: list

            teacher = Teacher()
            teacher.<caret>
            """.trimIndent(),
        )
        myFixture.completeBasic()
        val members = myFixture.lookupElementStrings ?: emptyList()
        // Delegated (name, greet), inherited-through-taker (serial), and role-native (courses).
        assertContainsElements(members, "name", "greet", "serial", "courses")
    }

    fun testNestedRoleDelegatesTransitively() {
        myFixture.configureByText(
            "nested_roles.py",
            """
            from __future__ import annotations
            from typing import Any, Generic, TypeVar

            T = TypeVar("T")

            class Role(Generic[T]):
                def __getattr__(self, name: str) -> Any: ...

            class Person:
                name: str

            class Teacher(Role[Person]):
                role_taker: Person
                courses: list

            class Department(Role[Teacher]):
                role_taker: Teacher
                budget: float

            dept = Department()
            dept.<caret>
            """.trimIndent(),
        )
        myFixture.completeBasic()
        val members = myFixture.lookupElementStrings ?: emptyList()
        // Nested role pulls in Teacher's own field and Person's delegated member.
        assertContainsElements(members, "name", "courses", "budget")
    }
}
