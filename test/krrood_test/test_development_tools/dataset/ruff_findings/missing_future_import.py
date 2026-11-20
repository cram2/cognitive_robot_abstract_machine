# reported
"""
A module without the import of postponed annotations.
"""

import os


def home_directory() -> str:
    """
    The home directory of the user running the process.
    """
    return os.path.expanduser("~")
