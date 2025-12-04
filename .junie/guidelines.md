# Junie Guidelines

## Testing
- If you need to run tests, execute them with pytest
- Reuse existing fixtures
- Always use a test-driven development approach

## Code Style
- Do not use abbreviations
- Create classes instead of using too many primitives
- Minimize duplication of code
- Do not wrap attribute access in try-except blocks
- Always access attributes via ".", never via getattr
- Use existing packages whenever possible
- Reduce nesting, reduce complexity
- Use short but descriptive names
- Always use dataclasses

## Design Principles
- Focus on strictly object oriented design
- Always apply the SOLID principles of object-oriented programming
- Create meaningful custom exceptions
- Eliminate YAGNI smells
- Make interfaces hard to misuse
- Reduce Nesting
- Reduce Complexity

## Documentation
- Write docstrings in ReStructuredText format 
- Write docstrings that explain what the function does and not how it does it
- Do not create type information for docstring
- Keep docstrings short and concise

## Misc
- If you find a package that could be replaced by a more powerful one, let us know