#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the GARMI clean-up demo with every grasp selection, and exit non-zero with a
    traceback on failure.
    """
    try:
        import demo

        for grasp_selection in demo.GraspSelection:
            demo.main(grasp_selection=grasp_selection)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
