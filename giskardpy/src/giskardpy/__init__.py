import os
import threading
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()


def preload_matplotlib():
    # preload costly imports that are not used immediately
    import matplotlib.pyplot
    from scipy import sparse
    import pandas


# Start preloading in the background
# if 'GITHUB_WORKFLOW' not in os.environ and not bool(getattr(__import__('sys').gettrace(), '__call__', None)):
#     threading.Thread(target=preload_matplotlib, daemon=True).start()
