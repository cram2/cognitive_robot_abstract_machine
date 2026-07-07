import os
import threading

from ._version import __version__


def preload_matplotlib():
    # preload costly imports that are not used immediately
    import matplotlib.pyplot
    from scipy import sparse
    import pandas


# Start preloading in the background
# if 'GITHUB_WORKFLOW' not in os.environ and not bool(getattr(__import__('sys').gettrace(), '__call__', None)):
#     threading.Thread(target=preload_matplotlib, daemon=True).start()
