# ANNarchy4 core
from .core.Global import *
from .core.Network import Network, MagicNetwork
from .core.IO import save, load, load_parameter
from .core.Neuron import RateNeuron, SpikeNeuron, IndividualNeuron
from .core.Synapse import RateSynapse, SpikeSynapse
from .core.Population import Population
from .core.PopulationView import PopulationView
from .core.Poisson import PoissonPopulation
from .core.Projection import Projection
from .core.Dendrite import Dendrite
from .core.Random import Constant, Uniform, Normal, LogNormal, Gamma, Exponential
from .core.Utils import raster_plot, smoothed_rate
try:
    from .core.cython_ext.Connector import CSR
except:
    _error('Could not import Cython modules. Try reinstalling ANNarchy.')

# ANNarchy compilation
from .generator import compile
from .extensions import *

# Generic imports
import numpy as np
import os, sys

# Version
__version__ = '4.1'
__release__ = '4.1.3'
core.Global._print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                   ') on ' + sys.platform + ' (' + os.name + ').' )
