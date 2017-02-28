# Generic imports
from __future__ import print_function
import os, sys
import numpy as np

# ANNarchy core
from .core.Global import *
from .core.Simulate import *
from .core.Neuron import Neuron
from .core.Synapse import Synapse
from .core.Population import Population
from .core.Projection import Projection
from .core.SpecificPopulation import PoissonPopulation, SpikeSourceArray, TimedArray, HomogeneousCorrelatedSpikeTrains
from .core.SpecificProjection import DecodingProjection
from .core.Dendrite import Dendrite
from .core.Random import Uniform, DiscreteUniform, Normal, LogNormal, Gamma, Exponential
from .core.IO import save, load, load_parameter, load_parameters, save_parameters
from .core.Utils import sparse_random_matrix
from .core.Monitor import Monitor
from .core.Network import Network, parallel_run
from .parser.report.Report import report
from .models import *
from .extensions import *

# Cython modules
try:
    # HD: until version 4.6 the connectivity class wasn't named properly. To ensure backward compability
    #     we rename the LILConnectivity to CSR
    from .core.cython_ext import LILConnectivity as CSR
except Exception as e:
    from .core.Global import _print
    _print(e)
    _print('Error: Could not import Cython modules. Try reinstalling ANNarchy.')

# ANNarchy compilation
from .generator import compile

# Version
__version__ = '4.6'
__release__ = '4.6.0'
core.Global._print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                    ') on ' + sys.platform + ' (' + os.name + ').' )
