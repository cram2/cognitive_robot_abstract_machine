#!/bin/bash
source /opt/ros/jazzy/setup.bash
cd ../examples
rm -rf tmp
mkdir tmp
jupytext --to notebook *.md
mv *.ipynb tmp
cd tmp
treon -v