"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from typing import Union

from ANNarchy.intern import Messages

# functions exported via wildcard import
__all__ = [
    # time-related
    'get_time', 'set_time', 'get_current_step', 'set_current_step'
]

class ConfigManager:
    """
    Manages the global configuration flags used in the ANNarchy framework. Users can manipulate this
    flags via two globally available functions:

        *setup()*
        *_optimization_flags()*

    Implementation Note:

        The class is implemented as singleton to ensure unique existance in the user space.
        One should not access the _config member directly but using the get_value_by_key() method.
    """
    _instance = None
    
    def __new__(self, *args, **kwds):
        """
        Only the first call will create a new instance of this class.
        """
        if self._instance is None:
            self._instance = super().__new__(self, *args, **kwds)
            self._config = dict(
                # Simulation Control
                dt = 1.0,
                # Logging
                verbose = False,
                suppress_warnings = False,
                show_time = False,
                # Performance-related
                disable_parallel_rng = True,
                use_seed_seq = True,
                use_cpp_connectors = False,
                disable_split_matrix = True,
                disable_SIMD_SpMV = True,
                disable_SIMD_Eq = False,
                # Datatype-related
                precision = "double",
                only_int_idx_type = True,
                # SpM formats
                sparse_matrix_format = "default",
                sparse_matrix_storage_order = "post_to_pre"
            )

            # This flags can not be configured through setup()
            self._performance_related_config_keys = [
                'disable_parallel_rng', 'use_seed_seq', 'use_cpp_connectors',
                'disable_split_matrix', 'disable_SIMD_SpMV', 'disable_SIMD_Eq'
            ]

        return self._instance

    def get_value_by_key(self, key: str) -> Union[str,float,bool]:
        """
        Returns the configuration for entry *key*. If the key does not
        exist a terminating exception is raised.
        """
        if key in self._config.keys():
            return self._config[key]
        else:
            raise Messages.ANNarchyException(key, "does not belong to global configuration keys.", exit=True)

    def set_value_by_key(self, key: str, value: Union[str,float,bool]):
        """
        Updates the configuration for entry *key* with a new *value*. 
        If the key does not exist a terminating exception is raised.
        """
        if key in self._config.keys():
            self._config[key] = value
        else:
            raise KeyError


#############################################
# Globally available functions (internal)
#############################################
def _optimization_flags(**keyValueArgs):
    """
    In particular the ANNarchy 4.7.x releases added various optional arguments to control the code generation. Please take in mind, that these
    flags might not being tested thoroughly on all features available in ANNarchy. They are intended for experimental features or performance analysis.

    * disable_parallel_rng: determines if random numbers drawn from distributions are generated from a single source (default: True). 
                            If this flag is set to true only one RNG source is used und the values are drawn by one thread which 
                            reduces parallel performance (this is the behavior of all ANNarchy versions prior to 4.7). 
                            If set to false a seed sequence is generated to allow usage of one RNG per thread. Please note, that this
                            flag won't effect the GPUs which draw from multiple sources anyways.
    * use_seed_seq: If parallel RNGs are used the single generators need to be initialized. By default (use_seed_seq == True) we use
                    the STL seed sequence to generate a list of seeds from the given master seed (*seed* argument). If set to False,
                    we use an improved version of the sequence generator proposed by M.E. O'Neill (https://www.pcg-random.org/posts/simple-portable-cpp-seed-entropy.html)
    * use_cpp_connectors:   For some of the default connectivity methods of ANNarchy we offer a CPP-side construction of the pattern to improve the
                            initialization time (default=False). For maximum performance the disable_parallel_rng should be set to False to allow
                            a parallel construction of the pattern.
    * disable_split_matrix: determines if projections can use thread-local allocation. If set to *True* (default) no thread local allocation is allowed.
                            This equals the behavior of ANNarchy until 4.7. If set to *False* the code generator can use sliced versions if they
                            are available.
    * disable_SIMD_SpMV: determines if the hand-written implementation is used (by default False) if the current hardware platform and used sparse matrix
                         format does support the vectorization). Disabling is intended for performance analysis.

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
        if key not in ConfigManager()._config.keys():
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
    return ConfigManager().get_value_by_key(key)

def _update_global_config(key: str, value: Union[str,float,bool]) -> None:
    """
    Updates a global configuration flag.

    Note: this function is intended for internal use.
          As user, please refer to *setup()* method.
    """
    print(key,value)
    return ConfigManager().set_value_by_key(key, value)
