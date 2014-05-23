# ANNarchy4 core
from .core.Global import *
from .core.Network import Network, MagicNetwork
from .core.IO import save, load, load_parameter
from .core.Neuron import RateNeuron, SpikeNeuron, IndividualNeuron
from .core.Synapse import RateSynapse, SpikeSynapse
from .core.Population import Population
from .core.PopulationView import PopulationView
from .core.Projection import Projection
from .core.Dendrite import Dendrite
from .core.Random import Constant, Uniform, Normal
from .core.Utils import raster_plot, smoothed_rate

# ANNarchy4 compilation
from .generator import compile

from .extensions import *

# Generic imports
import numpy as np
import os, sys

# Version
__version__ = '4.1'
__release__ = '4.1.1'
core.Global._print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                   ') on ' + sys.platform + ' (' + os.name + ').' )
