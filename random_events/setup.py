import ast

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "random_events_lib",  # Python module name
        ["src/random-events-lib/export/bindings.cpp",
         "src/random-events-lib/random_events_lib/src/sigma_algebra.cpp",
         "src/random-events-lib/random_events_lib/src/set.cpp",
         "src/random-events-lib/random_events_lib/src/product_algebra.cpp"],  # C++ binding source
        include_dirs=["src/random-events-lib/random_events_lib/include"],  # Include directory for C++ headers
        extra_compile_args=["-std=c++17", "-fPIC"],
    ),
]


def get_version():
    with open("src/random_events/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return ast.parse(line).body[0].value.s


setup(
    name="random_events",
    version=get_version(),
    author="Tom Schierenbeck",
    author_email="tom_sch@uni-bremen.de",
    description="Random random events for probabilistic reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomsch420/random-events",
    packages=["random_events"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
    ext_modules=ext_modules,
)