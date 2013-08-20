::
::  first we compile ANNarchyCore library.
cl build/*.cpp /DEBUG /EHsc /c /MD &&^
lib *.obj /verbose /OUT:ANNarchyCore.lib

::
::  first we compile ANNarchyCore library.
python setup.py build_ext -g --inplace
:: python setup.py build_ext --inplace