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
                show_time = False
            )

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

################################
# Globally available functions
################################
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
    return ConfigManager().set_value_by_key(key, value)
