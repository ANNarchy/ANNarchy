"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern import Messages

class GlobalObjectManager:
    """
    Helper class to ensure, that the created *Constant* objects are globally accessible.
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
        First call construction of the NetworkManager. No additional arguments are required.
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
            'constants': []
        }

    def clear(self):
        """
        Remove all instantiated constants.
        """
        for obj in self._objects['constants']:
            del obj

        self._create_initial_state()

    ################################
    ## Constants
    ################################
    def add_constant(self, new_constant):
        """
        Add a constant to the list.
        """
        # avoid doublons
        for obj in self._objects['constants']:
            if obj.name == new_constant.name:
                Messages._error('the constant', new_constant.name, 'is already defined.')
        # add to global list of constants
        self._objects['constants'].append(new_constant)

    def list_constants(self):
        """
        Returns a list of all constants declared with ``Constant(name, value)``.
        """
        l = []
        for obj in self._objects['constants']:
            l.append(obj.name)
        return l

    def get_constant(self, name):
        """
        Returns the ``Constant`` object with the given name, ``None`` otherwise.
        """
        for obj in self._objects['constants']:
            if obj.name == name:
                return obj
        return None

    def get_constants(self):
        return self._objects['constants']

    def number_constants(self):
        return len(self._objects['constants'])
