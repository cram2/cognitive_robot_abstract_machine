"""
Command line entry point of the DAiSy gripper demonstration::

    python3 -m experiments.daisy_gripper_demo --execution simulated
"""

from experiments.daisy_gripper_demo.demo import run_from_command_line

if __name__ == "__main__":
    run_from_command_line()
