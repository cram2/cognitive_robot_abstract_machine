#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the bullet world demo and exit non-zero with a traceback on failure.
    """
    try:
        import demo

        demo.main(event_segmentation=False)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
