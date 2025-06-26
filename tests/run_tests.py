"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
from ANNarchy import setup
from conftest import USED_PARADIGM, NUM_OMP_THREADS

# check command line arguments and configure ANNarchy
setup(num_threads = NUM_OMP_THREADS, paradigm = USED_PARADIGM)

# Different test-classes
from Common import *
from Interface import *
from Neuron import *
from RateSynapse import *
from SpikeSynapse import *

if __name__ == '__main__':

    # perform tests
    unittest.main(verbosity=2)
