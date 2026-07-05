package com.example.pyroles

import com.intellij.psi.PsiElement
import com.jetbrains.python.psi.types.PyClassMembersProviderBase
import com.jetbrains.python.codeInsight.PyCustomMember
import com.jetbrains.python.psi.PyCallExpression
import com.jetbrains.python.psi.PyClass
import com.jetbrains.python.psi.PyExpression
import com.jetbrains.python.psi.PyReferenceExpression
import com.jetbrains.python.psi.PySubscriptionExpression
import com.jetbrains.python.psi.PyTargetExpression
import com.jetbrains.python.psi.types.PyClassType
import com.jetbrains.python.psi.types.TypeEvalContext

/**
 * Teaches PyCharm about the **Role / delegation** pattern in Python.
 *
 * In that pattern a generic `Role[T]` wraps a "taker" instance (stored in a domain-named
 * field) and forwards attribute access to it through `__getattr__`, so a subclass gains the
 * taker's attributes by *composition* rather than inheritance:
 *
 * ```python
 * @dataclass
 * class Role(Generic[RoleTakerT]):
 *     def __getattr__(self, name: str) -> Any:
 *         return getattr(self.role_taker, name)
 *
 * @dataclass
 * class Person:
 *     name: str
 *     age: int
 *
 * @dataclass
 * class Teacher(Role[Person]):
 *     courses: list[str] = field(default_factory=list)
 *
 * teacher = Teacher(role_taker=Person("Ahmed", 20))
 * teacher.name   # <-- completion, type (str) and Go-to-Declaration now work
 * ```
 *
 * Because the access goes through `__getattr__`, PyCharm's static engine normally sees
 * nothing on `teacher` except its own fields. This provider reads the taker type from the
 * `Role[...]` base's generic argument and injects the taker's members onto the role's *type*.
 *
 * ### Why one hook is enough
 * Each injected [PyCustomMember] resolves to the **real** declaration inside the taker
 * class. Everything downstream — code completion, Go-to-Declaration (Ctrl/Cmd+Click),
 * Find Usages, Rename, and type inference (the member's type is read from its target) —
 * flows through that single resolve result. There is no need for separate completion and
 * navigation contributors.
 *
 * ### Scope handled
 * - **Direct roles** — `Teacher(Role[Person])`.
 * - **Transitive roles** — `SeniorTeacher(Teacher)` still receives `Person`'s members,
 *   because the `Role[...]` base is searched across the whole ancestry.
 * - **Inherited taker members** — if `Person(Entity)`, then `Entity`'s members are
 *   delegated too, because the taker's full MRO is walked.
 * - **Nested roles** — if the taker is *itself* a role (`VisitingTeacher(Role[Teacher])`),
 *   its delegated members are pulled in recursively.
 *
 * @see <a href="https://plugins.jetbrains.com/docs/intellij/pycharm.html">PyCharm plugin development</a>
 */
class RoleMembersProvider : PyClassMembersProviderBase() {

    private companion object {
        /**
         * Fully-qualified name of the krrood `Role` base class — a fast, precise match.
         *
         * This is only a fast path: when it does not match, [isRoleBase] falls back to a
         * structural check (the base defines a `__getattr__` method), so the plugin also
         * works for inline `Role[T]` bases such as the bundled `sample/roles_demo.py`.
         */
        const val ROLE_QUALIFIED_NAME: String = "krrood.patterns.role.Role"

        /** Dunder used to detect a delegating base in the structural fallback. */
        const val GETATTR: String = "__getattr__"

        /** The `typing.TypeVar` factory; used to read a `TypeVar(..., bound=...)` annotation. */
        const val TYPE_VAR: String = "TypeVar"

        /** Bases whose members are framework / builtin noise and must not be delegated. */
        val SKIPPED_TAKER_QNAMES: Set<String> = setOf(ROLE_QUALIFIED_NAME, "object", "typing.Generic")
    }

    /**
     * Called by the platform whenever it needs the members of a class *type* — for example
     * while completing `teacher.<caret>` or resolving the reference in `teacher.name`.
     *
     * @param classType the type the member access happens on
     * @param location  the reference site, when available
     * @param context   type-evaluation context to resolve against
     * @return the synthetic members contributed by role delegation (possibly empty)
     */
    override fun getMembers(
        classType: PyClassType,
        location: PsiElement?,
        context: TypeEvalContext,
    ): Collection<PyCustomMember> {
        // `Teacher.x` (the class object itself) is not delegation; only instances delegate.
        if (classType.isDefinition) return emptyList()

        // Insertion order is preserved; the first declaration seen (closest in the MRO) wins.
        val members = LinkedHashMap<String, PyCustomMember>()
        collectDelegatedMembers(classType.pyClass, context, members, HashSet())
        return members.values
    }

    /**
     * Adds into [out] every member that [roleClass] receives by delegation, following both
     * transitive role inheritance and nested roles. [visitedRoles] guards against cycles.
     */
    private fun collectDelegatedMembers(
        roleClass: PyClass,
        context: TypeEvalContext,
        out: MutableMap<String, PyCustomMember>,
        visitedRoles: MutableSet<PyClass>,
    ) {
        if (!visitedRoles.add(roleClass)) return

        for (taker in findTakers(roleClass, context)) {
            // 1. The taker's own members, plus everything it inherits.
            for (cls in selfAndAncestors(taker, context)) {
                addDeclaredMembers(cls, out)
            }
            // 2. If the taker is itself a role, pull in what *it* delegates, too.
            collectDelegatedMembers(taker, context, out, visitedRoles)
        }
    }

    /**
     * Returns the taker classes of [roleClass]: the `X` in every `Role[X]` base expression,
     * searched across the class itself and its whole ancestry (so a subclass of a role still
     * resolves the taker declared on its parent). The taker comes solely from the generic
     * argument, matching how `Role.get_role_taker_type` resolves it at runtime now that the
     * taker is the inherited `role_taker` field rather than a domain-named one.
     */
    private fun findTakers(roleClass: PyClass, context: TypeEvalContext): List<PyClass> {
        val takers = LinkedHashSet<PyClass>()
        for (cls in selfAndAncestors(roleClass, context)) {
            for (superExpr in cls.superClassExpressions) {
                extractTaker(superExpr, context)?.let(takers::add)
            }
        }
        return takers.toList()
    }

    /** If [superExpr] is a `Role[X]` base expression, resolves and returns `X`. */
    private fun extractTaker(superExpr: PyExpression, context: TypeEvalContext): PyClass? {
        val subscription = superExpr as? PySubscriptionExpression ?: return null
        val base = resolveClass(subscription.operand, context) ?: return null
        if (!isRoleBase(base, context)) return null
        return resolveClass(subscription.indexExpression, context)
    }

    /**
     * Decides whether [cls] is the `Role` base. Matches by [ROLE_QUALIFIED_NAME] first, then
     * falls back to a structural test: a role-like base delegates through a `__getattr__`
     * method (declared on the base itself or one of its ancestors).
     */
    private fun isRoleBase(cls: PyClass, context: TypeEvalContext): Boolean {
        if (cls.qualifiedName == ROLE_QUALIFIED_NAME) return true

        val family = listOf(cls) + cls.getAncestorClasses(context)
        return family.any { candidate ->
            candidate.methods.any { it.name == GETATTR }
        }
    }

    /**
     * Best-effort resolution of a type expression (a bare class name, an alias, or a bounded
     * `TypeVar`) to a [PyClass]. Tries reference resolution first — falling back to a
     * `TypeVar`'s bound when the reference is a type variable rather than a class — then
     * evaluates the expression's type.
     */
    private fun resolveClass(expr: PyExpression?, context: TypeEvalContext): PyClass? {
        if (expr == null) return null
        if (expr is PyReferenceExpression) {
            val resolved = expr.reference.resolve()
            (resolved as? PyClass)?.let { return it }
            resolveTypeVarBound(resolved, context)?.let { return it }
        }
        val type = context.getType(expr)
        return if (type is PyClassType && type.isDefinition) type.pyClass else null
    }

    /**
     * If [resolved] is a `T = TypeVar("T", bound=X)` target, resolves and returns the bound `X`.
     * krrood parameterises roles by such bounded type variables (`Role[TPerson]`), so the taker
     * is the bound rather than the type variable itself.
     */
    private fun resolveTypeVarBound(resolved: PsiElement?, context: TypeEvalContext): PyClass? {
        val target = resolved as? PyTargetExpression ?: return null
        val call = target.findAssignedValue() as? PyCallExpression ?: return null
        if ((call.callee as? PyReferenceExpression)?.referencedName != TYPE_VAR) return null
        val bound = call.getKeywordArgument("bound") ?: return null
        return resolveClass(bound, context)
    }

    /** [cls] followed by its full MRO, excluding `Role`, `object` and `Generic`. */
    private fun selfAndAncestors(cls: PyClass, context: TypeEvalContext): List<PyClass> =
        (listOf(cls) + cls.getAncestorClasses(context))
            .filter { it.qualifiedName !in SKIPPED_TAKER_QNAMES }

    /** Adds [cls]'s fields and methods to [out] (first-write-wins, noise filtered out). */
    private fun addDeclaredMembers(cls: PyClass, out: MutableMap<String, PyCustomMember>) {
        // Class-level fields, including dataclass / type-hinted attributes (`name: str`).
        for (attribute in cls.classAttributes) {
            offer(out, attribute.name, attribute)
        }
        // Attributes assigned on instances (`self.x = ...` in the body).
        for (attribute in cls.instanceAttributes) {
            offer(out, attribute.name, attribute)
        }
        // Methods, minus constructors and other dunders.
        for (method in cls.methods) {
            offer(out, method.name, method)
        }
    }

    /** Registers a single member if its name is publishable and not already taken. */
    private fun offer(out: MutableMap<String, PyCustomMember>, name: String?, target: PsiElement) {
        if (name == null || isDunder(name)) return
        out.putIfAbsent(name, PyCustomMember(name, target))
    }

    /** True for names like `__init__` / `__getattr__`; those are kept out of completion. */
    private fun isDunder(name: String): Boolean =
        name.length > 4 && name.startsWith("__") && name.endsWith("__")
}
