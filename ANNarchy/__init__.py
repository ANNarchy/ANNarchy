# ANNarchy core
from .core.Global import *
<<<<<<< HEAD
from .core.Network import Network, MagicNetwork
from .core.IO import save, load, load_parameter
from .core.Neuron import Neuron, RateNeuron, SpikeNeuron, IndividualNeuron
from .core.Synapse import Synapse, RateSynapse, SpikeSynapse
from .core.Population import Population
from .core.PopulationView import PopulationView
from .core.SpecificPopulation import PoissonPopulation, SpikeSourceArray
from .core.Projection import Projection
from .core.Dendrite import Dendrite
from .core.Random import Constant, Uniform, DiscreteUniform, Normal, LogNormal, Gamma, Exponential
from .core.Utils import raster_plot, smoothed_rate, histogram
from .models import *
=======
from .core.Neuron import Neuron, RateNeuron, SpikeNeuron
from .core.Synapse import Synapse, RateSynapse, SpikeSynapse
from .core.Population import Population
from .core.SpecificPopulation import PoissonPopulation, SpikeSourceArray
from .core.Projection import Projection
from .core.Dendrite import Dendrite
from .core.Random import Uniform, DiscreteUniform, Normal, LogNormal, Gamma, Exponential
from .core.IO import save, load
from .core.Utils import raster_plot, smoothed_rate, histogram
from .models import *

>>>>>>> v43/master
try:
    from .core.cython_ext.Connector import CSR
except:
    core.Global._error('Could not import Cython modules. Try reinstalling ANNarchy.')

# ANNarchy compilation
from .generator import compile
<<<<<<< HEAD
from .extensions import *
=======
#from .extensions import *
>>>>>>> v43/master

# Generic imports
import numpy as np
import os, sys

# Version
<<<<<<< HEAD
__version__ = '4.2'
__release__ = '4.2.4'
=======
__version__ = '4.3'
__release__ = '4.3.0'
>>>>>>> v43/master
core.Global._print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                   ') on ' + sys.platform + ' (' + os.name + ').' )
