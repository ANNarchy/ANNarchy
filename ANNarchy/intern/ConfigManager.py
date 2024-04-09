from ANNarchy.intern import Messages
from ANNarchy.intern.NetworkManager import NetworkManager

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
    
    _config = {
        'dt' : 1.0,
    }
    
    def __new__(self, *args, **kwds):
        if self._instance is None:
            self._instance = super().__new__(self, *args, **kwds)
        return self._instance

    def get_value_by_key(self, key):
        """
        Returns the configuration for entry *key*. If the key does not exist a terminating exception is raised.
        """
        if key in self._config.keys():
            return self._config[key]
        else:
            raise Messages.ANNarchyException(key, "does not belong to global configuration keys.", exit=True)

################################
# Globally available functions
################################
def get_global_config(key):
    return ConfigManager().get_value_by_key(key)
