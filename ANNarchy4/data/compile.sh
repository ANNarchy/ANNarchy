#!/bin/bash

export CFLAGS="-fopenmp -fpermissive -std=c++0x -L. -I. -Ibuild"

rm -f pyx/*.cpp
rm -f ANNarchyCore.so
rm -f libANNarchyCython.so

alone="False"

#check if cpp_stand_alone argument and update the value
if [ "${1%=*}" == "cpp_stand_alone" ]; then
    alone="${1#*=}"
fi

if [ $alone == "True" ]; then
    echo compile ANNarchy as stand alone library
    g++ -O2 -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCPP.so
    
else
    g++ -O2 -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCore.so

    python setup.py build_ext --inplace
fi



