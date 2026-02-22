---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hands-On Exercises

This section contains short, focused exercises that help you practice the concepts introduced in the User Guide. Each exercise is designed to be run as a notebook and to guide you from problem statement to a working solution.

What you will get
- Targeted practice that mirrors real usage of Semantic World
- Clear goals and constraints for each task
- Hints where appropriate, and space for your own solution

Prerequisites
- A working Python environment with the project dependencies installed: `pip install -r requirements.txt`
- Make sure to also install the self-assessment dependencies: `pip install -r requirements-self-assessment.txt`
- To check if you have set up everything correctly run `bash scripts/test_exercises.sh` from the project root.

How to use these exercises
1. Work through the corresponding topic in the User Guide first, so the terminology and workflows are familiar.
2. In your command line, navigate to scripts and run the command `bash scripts/convert_exercises_for_self_assessment.sh`
3. You will find the converted exercises inside the `self_assessment/exercises/converted_exercises` directory
4. Open the exercise notebook you want to work on and read the task description before touching any code.
5. Implement your solution in the dedicated cells. Keep your code small and readable. If your goal is to contribute to the project, consider also reading our [developer guide](https://cram2.github.io/semantic_world/developer_guide.html) first
6. Run the checks in the exercise to validate your work. If they pass, you may assume that your solution is correct.
7. If you are stuck or want to compare your solution to an example solution, you can find a working solution by coming back to this section, open the corresponding solution page on this documentation.

```{warning}
If you do mistakes while in a "world.modify_world()" context, or feel like your solutions should pass but they dont, you may have
accidentally altered the assumed starting world state. In this case, simply restart the kernel of your notebook and rerun all cells!
```

## Exercise Solutions

Below you may find solutions to the exercises.
- [](using-transformations-exercise)
- [](creating-custom-bodies-exercise)
- [](loading-worlds-exercise)
- [](visualizing-worlds-exercise)
- [](world-structure-manipulation-exercise)
- [](semantic-annotations-exercise)
- [](world-state-manipulation-exercise)
- [](regions-exercise)
