# Documentation
This directory contains the whole documentation for PyCRAM. To build the documentation please follow
the instructions below.



## Building the documentation


The documentation uses jupyter-book as engine.


Install the requirements in your python interpreter.

~~~
pip install -r requirements.txt
~~~
Run pycram and build the docs.

~~~
cd doc/source 
jupyter-book build .
~~~
Show the index.

~~~
firefox _build/html/index.html
~~~