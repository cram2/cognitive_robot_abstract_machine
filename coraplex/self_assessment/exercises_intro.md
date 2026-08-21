---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hands-On Exercises

This section contains exercises that let you practice writing plans with CoraPlex. Each one
is designed to be run as a notebook and to guide you from a problem statement to a robot that
carries it out.

What you will get
- Targeted practice that mirrors how a real demonstration is written
- Clear goals for each task, and checks that tell you whether you got there
- Space for your own solution, and an example solution to compare against

Prerequisites
- A working Python environment with the project dependencies installed: `pip install -r requirements.txt`
- The notebook tooling, which comes from the workspace's `doc` extra: `uv sync --extra doc`
  (this installs `jupytext`, `nbconvert`, `jupyter-book` and `treon`)
- A sourced ROS installation, including the workspace holding the robot descriptions. The
  exercises drive a PR2, whose model is resolved as `package://iai_pr2_description`.
- To check that everything is set up, run `bash scripts/test_exercises.sh` from the project root.

How to use these exercises
1. Work through the corresponding topic in the documentation first, so the terminology is familiar.
2. In your command line, navigate to the project root and run `bash scripts/convert_exercises_for_self_assessment.sh`
3. You will find the converted exercises inside the `self_assessment/exercises/converted_exercises` directory
4. Open the exercise notebook you want to work on and read the task description before touching any code.
5. Implement your solution in the dedicated cells. Keep your code small and readable.
6. Run the checks in the exercise to validate your work. If they pass, you may assume that your solution is correct.
7. If you are stuck or want to compare against an example solution, come back to this section and open the corresponding solution page.

```{warning}
The exercises carry one world forward from section to section: each step acts on the world the
step before it left behind. If a check fails even though your solution looks right, you may
have moved the robot or the objects in a way the next section does not expect. Restart the
kernel and rerun all cells from the top.
```

```{note}
These exercises really drive a simulated robot, so a full run takes minutes rather than
seconds. A section that seems to hang is usually just planning a motion.
```

## Exercise Solutions

Below you may find solutions to the exercises.
- [](writing-a-robot-plan-exercise)
