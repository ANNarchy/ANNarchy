Unittests for the ANNarchy neural simulation framework.

# Perform the tests

The unittests are executed using:

    pytest run_tests.py

By default, ANNarchy generates code for single-threaded execution. To verify the parallel implementations,
one need to run the test with an additional argument:

    pytest --openmp run_tests.py       --> openMP
    pytest --cuda run_tests.py         --> CUDA

# Code coverage

Note, in order to determine the code coverage, all three modes need to be run. Newer versions of code coverage
does not append anymore. One need to run the three sets of tests like the following:

coverage run --parallel-mode -m pytest run_tests.py
coverage run --parallel-mode -m pytest --openmp run_tests.py
coverage run --parallel-mode -m pytest --cuda run_tests.py

Afterwards the resulting .coverage* files can be combined using

coverage combine [filenames]

Then the report can be generated

coverage report -m
