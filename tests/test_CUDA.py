"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
from ANNarchy import setup

setup(paradigm="cuda")

from Common import *
from Interface import *
from Neuron import *
from RateSynapse import *
from SpikeSynapse import *


if __name__ == '__main__':
    unittest.main(verbosity=2)
