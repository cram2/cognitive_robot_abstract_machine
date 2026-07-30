#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the pouring demo and exit non-zero with a traceback on failure.
    """
    try:
        from experiments.tool_based_actions.simple_demo import demo_pouring

        demo_pouring.main()
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
