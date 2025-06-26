Unittests for the ANNarchy neural simulation framework.

The unittests are executed using:

    pytest run_tests.py

By default, ANNarchy generates code for single-threaded execution. To verify the parallel implementations,
one need to run the test with an additional argument:

    pytest --openmp run_tests.py       --> openMP
    pytest --cuda run_tests.py         --> CUDA

