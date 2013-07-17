#!/bin/bash

export CFLAGS="-fpermissive -std=c++0x -L. -I. -Ibuild"

rm -f pyx/*.cpp
rm -f ANNarchyCore.so
rm -f libANNarchyCython.so

g++ -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCore.so

python setup.py build_ext --inplace
