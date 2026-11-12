#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the HSR MuJoCo demo without a viewer and exit non-zero with a traceback on
    failure.
    """
    try:
        import demo

        demo.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
