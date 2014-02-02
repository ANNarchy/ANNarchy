#
# ANNarchy4 core
from .core.Global import *
from .core.Network import Network, MagicNetwork
from .core.IO import save, load, load_parameter
from .core.Neuron import RateNeuron, SpikeNeuron
from .core.Synapse import RateSynapse, SpikeSynapse
from .core.Population import Population
from .core.PopulationView import PopulationView
from .core.Projection import Projection
from .core.Dendrite import Dendrite
from .core.Connector import Connector, One2One, All2All, Gaussian, DoG
from .core.Random import Constant, Uniform, Normal

#
# ANNarchy4 visualizer
from .visualization import Visualization

#
# ANNarchy4 compilation
from .generator.Generator import compile

#
# extension packages, imported as available
from .extensions import *
if Global.config['verbose']:
    check_extensions()

# Generic imports
import numpy as np
import os, sys

# Version
__version__ = '4.1'
__release__ = '4.1.0.alpha'
Global._print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                   ') on ' + sys.platform + ' (' + os.name + ').' )
