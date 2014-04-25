#!/bin/bash

export CFLAGS="-D_DEBUG -fpermissive -std=c++0x -L. -I. -Ibuild"

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
    g++ -g -O0 -D_DEBUG -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCPP.so
    
else
    g++ -g -O0 -D_DEBUG -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.cpp -o libANNarchyCore.so

    python setup.py build_ext --inplace
fi > compile_stdout.log 2> compile_stderr.log



