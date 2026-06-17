# Code Quality Rules

## Avoid Behaviour
- Avoid using global variables
- avoid accessing any ormatic_interface.py files. if there are issues regarding the ormatic interface run the script `scripts/regenerate_all_orm.py`. If it does not fix the issue, consider consulting the developer.
- Avoid using mutable objects as default arguments

## Testing
- If you need to run tests, execute them with pytest
- Reuse existing fixtures found in conftest.py
- Always use a test-driven development approach. For example for bugs, always prove a bug by adding a meaningful, failing test first, before then fixing it
- When fixing failing tests, never modify the test itself
- All new features and fixes must be covered by tests

## Code Style
- Do not use abbreviations in variable names, methods, classes, or any other identifiers
- Method and class names should be concise and descriptive: they should tell *what* they do, not *how* they do it
- Create classes instead of using too many primitives. If a return type is always repeated, consider whether a dedicated class or type alias would convey more meaningful information
- Minimize duplication of code. Avoid placing methods in catch-all files like `utils.py`: prefer moving them onto a sensible class that owns the behaviour
- Comments must be meaningful and adhere to DRY; remove redundant or restating comments
- Do not wrap attribute access in try-except blocks
- Always access attributes via ".", never via getattr
- Use existing packages whenever possible
- Always use dataclasses
- Use short but descriptive names

## Imports
- Imports should always be absolute
- Imports should always be global (top of module), except in very special cases (for example ORM interface imports)
- Use stdlib type hints where possible, and for others use typing_extensions instead of typing
- Use `from __future__ import annotations` for type hints instead of wrapping the types in a string
- use TYPE_CHECKING guard for type-only imports

## Design Principles
- Focus on strictly object oriented design
- Always apply the SOLID principles of object-oriented programming
  - Single Responsibility: Each class/method does one thing. If a method has cyclomatic complexity in the hundreds, refactor.
  - Open/Closed: The code should be open for extension, but closed for modification.
  - Liskov Substitution: Subtypes must be substitutable for their base types without breaking behaviour.
  - Interface Segregation: Prefer many small interfaces over one large one.
  - Dependency Inversion: Depend on abstractions, not concrete implementations.
- Code should be modular and decoupled
- Create meaningful custom exceptions
- Eliminate YAGNI smells
- Make interfaces hard to misuse
- Reduce nesting and reduce complexity:
  - The main branch of a function should hold the main output with the biggest compute; alternative outputs should be realized via guard clauses beforehand
  - When dealing with nested if statements and branching methods, use guard clauses to reduce nesting by inverting conditions and returning early
- Avoid being too careful and overusing try-except blocks
- Instead of passing around strings, use enums instead

## Type Hints
- Classes and methods should always have accurate type hints (including `Any`) where applicable

## Documentation
- Classes and methods should always have meaningful, non-trivial documentation
- Write docstrings in ReStructuredText format
- Write docstrings that explain what the function does and not how it does it
- Keep docstrings short and concise
- Use Sphinx tags (for example `..note::`, `..warning::`, and `:func:`) where appropriate
- Do not create type information for docstrings (type hints already convey this)

## Domain-Specific Conventions
- When dealing with spatial types and connections, adhere to the style guide documented in `semantic_digital_twin/doc/style_guide.md`

## Misc
- If you find a package that could be replaced by a more powerful one, let us know
- Always use the Python interpreter that is set as the current project interpreter for running tests and commands
