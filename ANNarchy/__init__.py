# Generic imports
import os, sys
import numpy as np

# ANNarchy core
from .core.Global import *
from .core.Simulate import *
from .core.Constant import Constant
from .core.Neuron import Neuron
from .core.Synapse import Synapse
from .core.Population import Population
from .core.Projection import Projection
from .inputs import *
from .core.Dendrite import Dendrite
from .core.Random import Uniform, DiscreteUniform, Normal, LogNormal, Gamma, Exponential, Binomial
from .core.IO import save, load, load_parameter, load_parameters, save_parameters, MonitorList
from .core.Utils import sparse_random_matrix, sparse_delays_from_weights
from .core.Monitor import *
from .core.Network import Network, parallel_run
from .parser.report.Report import report
from .models.Neurons import *
from .models.Synapses import *
from .extensions import *
from .intern.ConfigManagement import setup

# Cython modules
try:
    # HD: until version 4.6 the connectivity class wasn't named properly. To ensure backward compability
    #     we rename the LILConnectivity to CSR
    from .cython_ext import LILConnectivity
    from .cython_ext import LILConnectivity as CSR
except Exception as e:
    print(e)
    print("""
Warning: Cython modules can not be imported. If you are installing ANNarchy, this is normal, ignore this message. If ANNarchy is already installed, something went wrong with the compilation, try reinstalling.
""")

# ANNarchy compilation
from .generator import compile

# several setup() arguments can be set on command-line
from ANNarchy.generator.CmdLineArgParser import CmdLineArgParser
_arg_parser = CmdLineArgParser()
_arg_parser.parse_arguments_for_setup()

# Automatically call ANNarchy.core.Global.clear()
# if the script terminates
import atexit
atexit.register(check_profile_results)
atexit.register(clear)

# Version
__version__ = '4.8'
__release__ = '4.8.2.6'

print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                    ') on ' + sys.platform + ' (' + os.name + ').' )
