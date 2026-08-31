# Documentation
This directory contains the whole documentation for CoraPlex. To build the documentation please follow
the instructions below.



## Building the documentation


The documentation uses jupyter-book as engine.


Install the documentation dependencies from the repository root.

~~~
pip install -e ".[doc]"
~~~
Run coraplex and build the docs.

~~~
cd doc/source 
jupyter-book build .
~~~
Show the index.

~~~
firefox _build/html/index.html
~~~