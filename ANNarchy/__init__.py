# Generic imports
import os, sys

# ANNarchy core
from .core.Global import magic_network, clear, check_profile_results
from .core.Constant import Constant
from .core.Neuron import Neuron
from .core.Parameters import Parameter, Variable, Creating, Pruning
from .core.Synapse import Synapse
from .core.Population import Population
from .core.Projection import Projection
from .inputs import *
from .core.Dendrite import Dendrite
from .core.Random import Uniform, DiscreteUniform, Normal, LogNormal, Gamma, Exponential, Binomial
from .core.IO import save, load, load_parameter, load_parameters, save_parameters, MonitorList
from .core.Utils import sparse_random_matrix, sparse_delays_from_weights, timeit
from .core.Monitor import Monitor
from .core.Network import Network
from .parser.report.Report import report
from .models.Neurons import *
from .models.Synapses import *
from .extensions import *

# Cython modules
try:
    # HD: until version 4.6 the connectivity class wasn't named properly. To ensure backward compability
    #     we rename the LILConnectivity to CSR
    from .cython_ext import LILConnectivity
    from .cython_ext import LILConnectivity as CSR
except Exception as e:
    print(e)
    print("""
-------------------------------------------------------------------------------------------------
Warning: Cython modules can not be imported. 
If you are installing ANNarchy, this is normal, ignore this message. 
If ANNarchy is already installed, something went wrong with the compilation, try reinstalling.
-------------------------------------------------------------------------------------------------
""")

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
__version__ = '5.0'
__release__ = '5.0.0rc4'

# Bad imports (from ANNarchy 4)
# compile, setup, simulate, simulate_until -> not used anymore
# reset, get_population, get_projection, populations, projections, monitors, enable_learning, disable_learning, get_time, set_time, get_current_step, set_current_step, dt, set_seed -> can now be directly imported from "core.Global"
def __getattr__(name):
    if name == "compile":
        raise ImportError(
            "The function 'compile' cannot be used anymore in ANNarchy 5.\n"
            "Please update your code to use the compile() function of a Network object.\n"
            "In case you defined populations, projections, and monitors as in ANNarchy 4, you can obtain the corresponding network object using the 'magic_network()' function.\n"
        )
    elif name == "setup":
        raise ImportError(
            "The function 'setup' cannot be used anymore in ANNarchy 5.\n"
            "Please update your code to use the config() function of a Network object.\n"
        )
    elif name == "simulate":
        raise ImportError(
            "The function 'simulate' cannot be used anymore in ANNarchy 5.\n"
            "Please update your code to use the simulate() function of a Network object.\n"
        )
    elif name == "simulate_until":
        raise ImportError(
            "The function 'simulate_until' cannot be used anymore in ANNarchy 5.\n"
            "Please update your code to use the simulate_until() function of a Network object.\n"
        )
    elif name in ["reset", "get_population", "get_projection", "populations", "projections", "monitors",
                  "enable_learning", "disable_learning", "get_time", "set_time", "get_current_step",
                  "set_current_step", "dt", "set_seed"]:
        raise ImportError(
            f"The function '{name}' should not be used anymore in ANNarchy 5.\n"
            "Please update your code to use the corresponding function of a Network object.\n"
            "In case you defined a magic network (function 'magic_network') in ANNarchy 4 style, you can still import the function '{name}' from the 'ANNarchy.core.Global' module.\n"
        )
    raise AttributeError(f"module 'ANNarchy' has no attribute '{name}'")

print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                    ') on ' + sys.platform + ' (' + os.name + ').' )
