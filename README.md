# Cognitive Robot Abstract Machine (CRAM)

Monorepo for the CRAM cognitive architecture. 

## Installation 

To install the CRAM architecture, follow these steps:

Setup the Python venvironment:

```bash
sudo apt install -y virtualenv virtualenvwrapper && \
grep -qxF 'export WORKON_HOME=$HOME/.virtualenvs' ~/.bashrc || echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc && \
grep -qxF 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' ~/.bashrc || echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc && \
grep -qxF 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' ~/.bashrc || echo 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' >> ~/.bashrc && \
source ~/.bashrc && \
mkvirtualenv cram-env
```
Activate / deactivate

```
workon cram-env
deactivate
```


We use poetry to manage dependencies. Install poetry if you haven't already:

```bash
pip install poetry
``` 

Install the CRAM package along with its dependencies:

```bash
cd cognitive_robot_abstract_machine
git submodule update --init --recursive
poetry install
```

## Contribution

Before committing any changes, please navigate into the project root and install pre-commit hooks:

```bash
sudo apt install pre-commit
pre-commit install
```

### Code of Conduct

> Any code added to the repository must have at least an 85% test coverage.

🚀 How to Create a Pull Request (PR) in Our System

This guide outlines the best practices for creating a Pull Request in our system to ensure high-quality, maintainable, and robust code.

1. 🤖 AI Code Review First (Pre-PR Check)

    Before submitting your PR, let AI review your code to catch common issues and suggest improvements.

    - GitHub Copilot Reviewer: You can integrate GitHub Copilot as a reviewer directly into your PR process.

    - PyCharm Integration: Alternatively, use AI features directly within PyCharm for an immediate local review.

2. 🛡️ Embrace Test-Driven Thinking

    Our process is Test-Driven Development (TDD):

    - Bug Fixes: If you find a bug, your first step is to write a test that fails (reproduces the bug). Push this failing test, then implement the fix, and ensure the test now passes.

    - The Beyoncé Rule: "If you like it, you should put a test on it." Every new feature or piece of logic needs corresponding tests.

3. 🎯 Focus on Method Quality and Complexity

    - Method Length/Complexity: Keep your methods concise and focused. If the cyclomatic complexity (which you can check using plugins like Code Complexity for JetBrains) reaches the hundreds, your method is highly likely to need refactoring/improvement.

    - Helper Methods: Extract duplicate code into helper methods to adhere to the Don't Repeat Yourself (DRY) principle.

    - Modularity/Plugin Thinking: Think modularly. Code should be designed with a plugin-like approach, making components easily interchangeable or extendable.

    - Side Effects & Entanglement: Methods should be decoupled (not entangled) and should not have hidden side effects. They should ideally do one thing and do it well, following the Single Responsibility Principle (SRP) from SOLID.
Example:
        ```python
        counter = 0      # outer scope state

        def increment_counter():
            global counter
            counter += 1   # side effect: modifies outer scope state
            print(f"Counter is now {counter}")  # side effect: I/O
        ```

4. 📐 Adhere to Code Style and Principles

    - Code Formatting: We use Black, which is fully PEP 8 compliant. All code must be automatically formatted with Black before submission.

    - SOLID Principles: Read and understand the SOLID principles for writing robust, maintainable, and scalable software. This article is a great resource: https://realpython.com/solid-principles-python/

5. ✍️ Naming, Typing, and Imports

    - Descriptive Naming: Choose descriptive names for variables, functions, and classes. Names should clearly communicate intent.

    - Correct Typing: Use correct type hints consistently throughout your code.

    Import Strategy:

    - Use relative importing (e.g., from . import utils) always within the package.

    - When importing types, use typing extensions instead of typing or the standard library types;

    - Avoid importing types directly from modules if you don't need to construct an instance of the type inside the module. Annotations can be imported with the TYPE_CHECKING guard.

6. 📝 Documentation and Comments

    - Non-Trivial Code: Document everything that is not trivial to understand.

    - Avoid Over-Swaffling: Be concise. Do not use unnecessary, verbose explanations.

    - Inline Comments: Use inline comments sparingly, primarily for explaining complex logic or a long block of code.

7. 🔁 Final Review and Responsibility

    - Human Review: Always perform a final human review of your own code before submitting the PR. Read it line-by-line as if you were the reviewer.

    - "If you break it, you fix it" Rule: You are the primary owner and person responsible for the code you introduce. If a bug is found in your changes, you must prioritize its fix.

    - The "One Change, One Commit" Rule: Each commit should be a logical, atomic unit of work.

8. Post-Submission and Review 🔎

    After you open your Pull Request (PR), it enters the review stage. This is a critical step for ensuring code quality and collaboration.

    Responding to Feedback:

    - Mindset is Key: Remember that code reviews are about the code, not you. Feedback is given to improve the project's quality and help you learn. Take all comments professionally and constructively.

     - Addressing the Feedback: When a reviewer requests changes, you don't need to close the PR and start over! Simply make the required modifications in your local working directory.

    - Commit and Push: Once the changes are made, create a new commit and push it to the same feature branch you used for the PR. The PR will automatically update with your new commits.
                
        ```
        # 1. Make the changes locally...
        git add .
        git commit -m "Address review feedback on component X"
        git push origin <your-feature-branch-name>
        ```

PR Checklist Summary

    [ ] AI (Copilot/PyCharm) has reviewed the code.

    [ ] Black has formatted the code (PEP 8 compliant).

    [ ] New features/logic have tests (Beyoncé Rule).

    [ ] Bug fixes include a test that reproduced the bug.

    [ ] Methods are concise and low in complexity.

    [ ] Descriptive names and correct type hints are used.

    [ ] Relative importing is used correctly.

    [ ] Non-trivial code is documented concisely.

    [ ] Code is modular, decoupled, and adheres to SOLID principles.

    [ ] Final personal human review complete.