::
::  first we compile ANNarchyCore library.
cl build/*.cpp /EHsc /c /MD &&^
lib *.obj /verbose /OUT:ANNarchyCore.lib

::
::  first we compile ANNarchyCore library.
python setup.py build_ext --inplace
