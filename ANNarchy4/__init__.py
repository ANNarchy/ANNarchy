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
from .core.Variable import Variable, SpikeVariable

#
# ANNarchy4 visualizer
from .visualization import Visualization

try:
    import PyQt4
    import OpenGL
    from .editor import ANNarchyEditor
except ImportError:
    print "ANNarchy4 editor is not available through missing dependencies."

#
# ANNarchy4 compilation
from .generator import compile

#
# extension packages, imported as available
from .extensions import *
if core.Global.config['verbose']:
    check_extensions()

import numpy as np
__version__ = '4.1'
__release__ = '4.1.0.alpha'
