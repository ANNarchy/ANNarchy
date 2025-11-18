# Generic imports
import os, sys
import types

# ANNarchy core
from .core.Global import *
from .core.Simulate import *
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
from .intern.ConfigManagement import setup
from .intern import Messages

# Cython modules
try:
    from .cython_ext import LILConnectivity
except Exception as e:
    print(e)
    print("""
-------------------------------------------------------------------------------------------------
Warning: Cython modules can not be imported.
If you are installing ANNarchy, this is normal, ignore this message.
If ANNarchy is already installed, something went wrong with the compilation, try reinstalling.
-------------------------------------------------------------------------------------------------
""")

# Compilation (like in ANNarchy 4)
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
__version__ = '5.0'
__release__ = '5.0.0rc10'

# Deprecated imports from this module:
_deprecated = {
    "Population", "Projection", "Monitor", "compile", "setup", "simulate", "simulate_until",
    "reset", "get_population", "get_projection", "populations", "projections", "monitors",
    "enable_learning", "disable_learning", "get_time", "set_time", "get_current_step",
    "set_current_step", "dt", "set_seed", "step", "callbacks_enabled", "disable_callbacks",
    "enable_callbacks", "clear_all_callbacks", "save", "load", "save_parameters", "load_parameters",
}
def _deprecated_wrapper(obj, name, message=None):
    if isinstance(obj, types.FunctionType):
        def wrapper(*args, **kwargs):
            Messages._warning(message or f"{name} is deprecated. Please use ann.Network().{name}() instead.")
            return obj(*args, **kwargs)
        wrapper.__name__ = name
        wrapper.__doc__ = obj.__doc__
        return wrapper
    elif isinstance(obj, type):  # class
        class DeprecatedClass(obj):
            def __new__(cls, *args, **kwargs):
                Messages._warning(message or f"{name} is deprecated. Please use ann.Network().{name}() instead.")
                return super().__new__(cls)
        DeprecatedClass.__name__ = name
        DeprecatedClass.__doc__ = obj.__doc__
        return DeprecatedClass
    else:
        return obj
    
for name in _deprecated:
    obj = globals().get(name)
    if obj is not None:
        if name == "Population":
            message = "Population is deprecated. Please use ann.Network().create() instead."
        elif name == "Projection":
            message = "Projection is deprecated. Please use ann.Network().connect() instead."
        elif name == "Monitor":
            message = "Monitor is deprecated. Please use ann.Network().monitor() instead."
        elif name == "compile":
            message = "compile is deprecated. Please use ann.Network().compile() instead.\n\t Since ANNarchy 5.0 the compilation of shadow networks is not supported anymore\n\t and using them possibly could generate errorneous simulation results or crash.\n\t You need to construct a Network object, please refer to the documentation."
        elif name == "set_seed":
            message = "set_seed is deprecated. Please use ann.Network().seed() instead."
        elif name == "save_parameters":
            message = "save_parameters is deprecated and will be removed in future releases."
        elif name == "load_parameters":
            message = "load_parameters is deprecated. Parameters can be directly loaded and used as dictionaries in Neuron and Synapse definitions."
        else:
            message = None
        globals()[name] = _deprecated_wrapper(obj, name, message)



print( 'ANNarchy ' + __version__ + ' (' + __release__ + \
                    ') on ' + sys.platform + ' (' + os.name + ').' )
