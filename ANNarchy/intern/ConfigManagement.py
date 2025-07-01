"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from typing import Union
import numpy as np
import copy

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern import Messages

default_config = dict(
    # Simulation Control
    dt = 1.0,
    seed = None,
    method = 'explicit',
    structural_plasticity = False,
    # Parallel processing
    num_threads = 1,
    visible_cores = [],
    paradigm = 'openmp',
    # Datatype-related
    precision = "double",
    only_int_idx_type = True,
    # Logging
    verbose = False,
    suppress_warnings = False,
    show_time = False,
    trace_calls = None,
    # Performance-related
    disable_parallel_rng = True,
    use_seed_seq = True,
    use_cpp_connectors = False,
    disable_split_matrix = True,
    disable_SIMD_SpMV = True,
    disable_SIMD_Eq = False,
    disable_bitmask = True,
    # SpM formats
    sparse_matrix_format = "default",
    sparse_matrix_storage_order = "post_to_pre",
    # Profiling
    profiling = False,
    profile_out = None,
    # Other
    debug = False,
    disable_shared_library_time_offset = True
)

class ConfigManager:
    """
    Manages the global configuration flags used in the ANNarchy framework. Users can manipulate this
    flags via two globally available functions:

        *setup()*
        *_optimization_flags()*

    Implementation Note:

        The class is implemented as singleton to ensure unique existance in the user space.
        One should not access the _config member directly but using the get/set methods.
    """
    _instance = None
    
    def __new__(self, *args, **kwds):
        """
        Only the first call will create a new instance of this class.
        """
        if self._instance is None:
            self._instance = super().__new__(self, *args, **kwds)
            self._config = { # dictionary because network ids can change
                0: copy.deepcopy(default_config)
            }

            # This flags can not be configured through setup()
            self._performance_related_config_keys = [
                'use_cpp_connectors', 'disable_split_matrix', 'disable_SIMD_SpMV', 'disable_SIMD_Eq'
            ]

        return self._instance
    
    def register_network(self, net_id:int=0):
        """
        Registers a network by copying the "magic" configuration of id 0.
        """
        self._config[net_id] = copy.deepcopy(self._config[0])
    
    def get_config(self, net_id:int=0):
        """
        Returns the config for the given network.
        """
        return copy.deepcopy(self._config[net_id])
    
    def set_config(self, net_id:int, config):
        """
        Set the config for the given network.
        """
        if not net_id in self._config.keys():
            self.register_network(net_id)
        self._config[net_id].update(config)

    
    def get(self, key: str, net_id:int=0) -> str | float | bool:
        """
        Returns the configuration for entry `key` in the network `net_id`. 
        
        If the key does not exist, a terminating exception is raised.
        """
        if key in self.keys():
            return self._config[net_id][key]
        else:
            raise Messages.ANNarchyException(key, "does not belong to global configuration keys.", exit=True)
        

    def set(self, key: str, value: str | float | bool, net_id:int=0):
        """
        Updates the configuration for entry *key* with a new *value*. 
        If the key does not exist a terminating exception is raised.
        """
        if key in self.keys():
            self._config[net_id][key] = value
        else:
            raise KeyError
        
    def keys(self) -> list:
        "Returns the list of keys that can be set with setup."
        return list(self._config[0].keys())


#############################################
# Globally available functions (internal)
#############################################
def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy simulations. 
    
    It takes various optional arguments. The most useful ones are:

    * `dt`: simulation step size in milliseconds (default: 1.0).
    * `paradigm`: parallel framework for code generation. Accepted values: "openmp" or "cuda" (default: "openmp").
    * `method`: default method to numerize ODEs. Default is the explicit forward Euler method ('explicit').
    * `precision`: default floating precision for variables in ANNarchy. Accepted values: "float" or "double" (default: "double")
    * `structural_plasticity`: allows synapses to be dynamically added/removed during the simulation (default: False).
    * `seed`: the seed (integer) to be used in the random number generators (default = None is equivalent to time(NULL)).
    * `num_threads`: number of treads used by openMP (overrides the environment variable ``OMP_NUM_THREADS`` when set, default = None).

    Flags related to the optimization of the simulation kernels are:

    * `sparse_matrix_format`: the default matrix format for projections in ANNarchy (by default: List-In-List for CPUs and Compressed Sparse Row). Note that this affects only the C++ data structures.
    * `sparse_matrix_storage_order`: encodes whether the row in a connectivity matrix encodes pre-synaptic neurons (post_to_pre, default) or post-synaptic neurons (pre_to_post). Note that affects only the C++ data structures.
    * `only_int_idx_type`: if set to True (default) only signed integers are used to store pre-/post-synaptic ranks which was default until 4.7.If set to False, the index type used in a single projection is selected based on the size of the corresponding populations.
    * `visible_cores`: allows a fine-grained control which cores are useable for the created threads (default = [] for no limitation). It can be used to limit created openMP threads to a physical socket.

    The following parameters are mainly for debugging and profiling, and should be ignored by most users:

    * `verbose`: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    * `suppress_warnings`: if True, warnings (e. g. from the mathematical parser) are suppressed.
    * `show_time`: if True, initialization times are shown. Attention: verbose should be set to True additionally.
    * `disable_shared_library_time_offset`: by default False. If set to True, the shared library generated by ANNarchy will not be extended by time offset.

    **Note:**

    This function should be used before any other functions of ANNarchy (including importing a network definition), right after `import ANNarchy`:

    ```python
    import ANNarchy as ann
    ann.setup(dt=1.0, method='midpoint', num_threads=2)
    ```

    """
    if len(NetworkManager().get_network(0).get_populations()) > 0:
        if 'dt' in keyValueArgs:
            Messages._warning('setup(): populations or projections have already been created at the global level. Changing dt now might lead to strange behaviors with the synaptic delays (internally generated in steps, not ms)...')
        if 'precision' in keyValueArgs:
            Messages._warning('setup(): populations or projections have already been created at the global level. Changing precision now might lead to strange behaviors...')

    config_manager = ConfigManager()

    for key in keyValueArgs:
        # sanity check: filter out performance flags
        if key in ConfigManager()._performance_related_config_keys:
            Messages._error("Performance related flags can not be configured by setup()")

        if key in ConfigManager().keys():
            ConfigManager().set(key, keyValueArgs[key], net_id=0)
        else:
            Messages._warning('setup(): unknown key:', key)

        if key == 'seed': # also seed numpy, but this is the old way
            np.random.seed(keyValueArgs[key])

        if key == 'sparse_matrix_format':
            # check if this is a supported format
            if keyValueArgs[key] not in ["lil", "csr", "csr_vector", "csr_scalar", "dense", "ell", "ellr", "sell", "coo", "bsr", "hyb", "auto"]:
                Messages._error("The value", keyValueArgs[key], "provided to sparse_matrix_format is not valid.")

def _optimization_flags(**keyValueArgs):
    """
    In particular the ANNarchy 4.7.x releases added various optional arguments to control the code generation. Please take in mind, that these flags might not being tested thoroughly on all features available in ANNarchy. They are intended for experimental features or performance analysis.

    * disable_parallel_rng: determines if random numbers drawn from distributions are generated from a single source (default: True).  If this flag is set to true only one RNG source is used und the values are drawn by one thread which reduces parallel performance (this is the behavior of all ANNarchy versions prior to 4.7). If set to false a seed sequence is generated to allow usage of one RNG per thread. Please note, that this lag won't effect the GPUs which draw from multiple sources anyways.
    * use_seed_seq: If parallel RNGs are used the single generators need to be initialized. By default (use_seed_seq == True) we use the STL seed sequence to generate a list of seeds from the given master seed (*seed* argument). If set to False, we use an improved version of the sequence generator proposed by M.E. O'Neill (https://www.pcg-random.org/posts/simple-portable-cpp-seed-entropy.html)
    * use_cpp_connectors:   For some of the default connectivity methods of ANNarchy we offer a CPP-side construction of the pattern to improve the initialization time (default=False). For maximum performance the disable_parallel_rng should be set to False to allow a parallel construction of the pattern.
    * disable_split_matrix: determines if projections can use thread-local allocation. If set to *True* (default) no thread local allocation is allowed. This equals the behavior of ANNarchy until 4.7. If set to *False* the code generator can use sliced versions if they are available.
    * disable_SIMD_SpMV: determines if the hand-written implementation is used (by default False) if the current hardware platform and used sparse matrix format does support the vectorization). Disabling is intended for performance analysis.

    * disable_SIMD_Eq: this flags disables auto-vectorization and openMP simd (by default False). Disabling is intended for performance analysis.

    **Note:**

    This function should be used only for special purposes therefore its not publicly available.

    ```python
    from ANNarchy import *  # will not work
    from ANNarchy.core.Global import _optimization_flags
    _optimization_flags(disable_parallel_rng=False)
    ```

    """
    for key in keyValueArgs:
        # Sanity check: valid key?
        if key not in ConfigManager().keys():
            Messages._warning('_optimization_flags() received unknown key:', key)
            continue

        if key not in ConfigManager()._performance_related_config_keys:
            Messages._warning(f"The key '{key}' does not belong to the performance related keys.")
            continue

        # Update global config
        _update_global_config(key, keyValueArgs[key])

        # Warning: use_cpp_connectors and disable_parallel_rng should be both activated
        if key == "use_cpp_connectors":
            if get_global_config(key) == True:
                Messages._warning("use_cpp_connectors is an experimental feature, we greatly appreciate bug reports.")

                # check if the key is in the update list
                if 'disable_parallel_rng' in keyValueArgs.keys():
                    if keyValueArgs['disable_parallel_rng']:
                        Messages._warning("If 'use_cpp_connectors' is enabled, the 'disable_parallel_rng' flag should be disabled for maximum efficiency.")
                # is it enabled by default?
                elif get_global_config('disable_parallel_rng'):
                    Messages._warning("If 'use_cpp_connectors' is enabled, the 'disable_parallel_rng' flag should be disabled for maximum efficiency.")
                # no conflict
                else:
                    pass

#############################################
# Globally available functions (internal)
#############################################
def get_global_config(key: str) -> Union[str,float,bool]:
    """
    Returns a global configuration.
    """
    return ConfigManager().get(key, net_id=0)

def _update_global_config(key: str, value: Union[str,float,bool]) -> None:
    """
    Updates a global configuration flag.

    Note: this function is intended for internal use.
          As user, please refer to *setup()* method.
    """
    return ConfigManager().set(key, value, net_id=0)

def _check_paradigm(paradigm, net_id=0):
    """
    Returns True when the provided paradigm is currently used.

    Possible values:

    1. "openmp"
    2. "cuda"
    """
    try:
        return paradigm == ConfigManager().get('paradigm', net_id)
    except KeyError:
        Messages._error("Unknown paradigm")

def _check_precision(precision, net_id=0):
    """
    Returns True when the provided precision is currently used.

    Possible values:

    1. "float"
    2. "double"
    """
    try:
        return precision == ConfigManager().get('precision', net_id)
    except KeyError:
        Messages._error("Unknown precision")
