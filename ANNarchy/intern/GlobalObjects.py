"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern import Messages

class GlobalObjectManager:
    """
    Helper class to ensure, that the neurons, synapses and functions are globally accessible.
    """
    # singleton instance
    _instance = None
    # storage
    _objects = None

    def __init__(self):
        """
        Constructor.
        """
        pass

    def __new__(cls):
        """
        First call construction of the GlobalObjectManager. No additional arguments are required.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._create_initial_state(cls)
        
        return cls._instance

    def _create_initial_state(self):
        """
        Initialize the container for globally available objects.

        Called either from __new__ or clear(). The first
        slot is reserved for the magic network.
        """
        self._objects = {
            'neurons': [],
            'synapses': [],
            'functions': []
        }

    def clear(self, functions:bool=True, neurons:bool=True, synapses:bool=True):
        """
        Remove all instantiated objects.
        """
        if functions:
            self._objects['functions'] = []

        if neurons:
            for obj in self._objects['neurons']:
                del obj
            self._objects['neurons'] = []

        if synapses:
            for obj in self._objects['synapses']:
                del obj
            self._objects['synapses'] = []

    ################################
    ## Neuron types
    ################################
    def add_neuron_type(self, neuron):
        self._objects['neurons'].append(neuron)

    def get_neuron_types(self):
        return self._objects['neurons']

    def num_neuron_types(self):
        return len(self._objects['neurons'])

    ################################
    ## Synapse types
    ################################
    def add_synapse_type(self, synapse):
        self._objects['synapses'].append(synapse)

    def get_synapse_types(self):
        return self._objects['synapses']

    def num_synapse_types(self):
        return len(self._objects['synapses'])


    ################################
    ## Functions
    ################################
    def add_function(self, function):
        name = function.split('(')[0]
        if self.get_function(name) is not None:
            Messages._error(f"add_function(): the global function {name} already exists at the global level.")
        self._objects['functions'].append( (name, function) )

    def functions(self, name:str, network=None):

        net_id = 0 if network is None else network.id
        try:
            func = getattr(NetworkManager().get_network(net_id=net_id).instance, 'func_' + name)
        except:
            Messages._error('call to', name, ': the function is not compiled yet.')

        return func

    def get_function(self, name):
        """
        Returns the function string with the provided name.
        """
        for n, function in self._objects['functions']:
            if n == name:
                return name
        return None

    def get_functions(self):
        """
        Returns all functions
        """
        return self._objects['functions']

    def number_functions(self):
        return len(self._objects['functions'])


    ################################
    ## Constants
    ################################

    def list_constants(self):
        """
        Only for the parser, constants are stored in networks
        """
        names = []
        for network in NetworkManager().get_networks():
            if network is None:
                # The network was either already deleted or not completely
                # initialized ...
                continue

            constants = network.get_constants()
            for c in constants:
                names.append(c.name)

        return list(set(names))
