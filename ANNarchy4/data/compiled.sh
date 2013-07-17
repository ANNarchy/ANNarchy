#!/bin/sh

export CFLAGS="-D_DEBUG -fpermissive -std=c++0x -L. -I. -Ibuild"

rm -f pyx/*.cpp
rm -f ANNarchyCore.so

g++ -D_DEBUG -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCore.so

python setup.py build_ext --inplace
