"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
# Equals the ANNarchy default configuration, i.e. single-thread simulation.
NUM_OMP_THREADS = 1
USED_PARADIGM = 'openmp'

def pytest_addoption(parser):
    """
    Extend the default arguments of pytest.
    """
    parser.addoption(
        "--openmp",
        action="store_true",
        default=False,
        help="TODO"
    )
    parser.addoption(
        "--cuda",
        action="store_true",
        default=False,
        help="TODO"
    )

def pytest_configure(config):
    """
    Updates the runtime configuration if necessary. Automatically called by pytest.
    """
    global NUM_OMP_THREADS, USED_PARADIGM

    if config.getoption("--cuda"):
        USED_PARADIGM = "cuda"
        NUM_OMP_THREADS = 1
    elif config.getoption("--openmp"):
        USED_PARADIGM = "openmp"
        NUM_OMP_THREADS = 3

    print(f"[TEST CONFIGURATION] paradigm={USED_PARADIGM}, num_threads={NUM_OMP_THREADS}")