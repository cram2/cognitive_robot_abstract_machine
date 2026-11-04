#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the DAiSy gripper demo and exit non-zero with a traceback on failure.

    The demo is skipped when DAiSy is not installed.
    """
    try:
        from semantic_digital_twin.utils import daisy_installed

        if not daisy_installed():
            print("DAiSy is not installed, skipping the gripper demo.")
            return

        from experiments.daisy_gripper_demo import demo

        demo.main()
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
