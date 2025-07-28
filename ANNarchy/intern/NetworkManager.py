"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import os
import shutil
import time
import random
import string

from ANNarchy.intern import ConfigManagement
from ANNarchy.intern import Messages


class IDGenerator:
    """
    Returns random IDs that are guaranteed to be unique.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IDGenerator, cls).__new__(cls)
            cls._instance.current_id = 0
        return cls._instance

    def generate_ID(self):
        """Generate a unique ID."""
        new_id = self.current_id
        self.current_id += 1
        return new_id


class NetworkManager :
    """
    This class implements the management of the different networks. The
    add/remove methods allow the 'random' removal of the network instances and later
    refill of empty spaces.

    The class is implemented as singleton and therefore initialized on first request.

    Individual network are accessed by their id and allow the access to their components.
    """
    _instance = None    # singleton instance

    def __new__(cls):
        """
        First call construction of the NetworkManager. No additional arguments are required.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize(cls._instance)
        
        return cls._instance

    def _initialize(self) -> None:
        """
        Initialize the container for the initial network.

        Called either from __init__ or clear(). The first
        slot net_id = 0 is reserved for the magic network.
        """
        # Initialize the list of networks woth the magic network
        self._networks = []
        from ANNarchy.core.Network import Network
        magic_network = Network()

    def get_id(self):
        """
        Returns a unique integer ID that can be safely used to name populations or projections over different networks.
        """
        # ID generator
        generator = IDGenerator()
        return generator.generate_ID()


    def add_network(self, net):
        """
        Adds an empty structure for a new network and returns the new network ID.
        """
        new_id = len(self._networks)
        self._networks.append(net)

        Messages._debug("NetworkManager: added network " + str(net) + " and assigned ID = " + str(new_id))

        return new_id

    def get_network(self, net_id:int) -> "Network":
        "Returns the network with the corresponding id."
        if net_id < len(self._networks):
            return self._networks[net_id]
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_networks(self) -> list["Network"]:
        "Returns the list of networks."
        return self._networks

    def __getitem__(self, net_id:int):
        return self._networks[net_id]
    
    def magic_network(self) -> "Network":
        "Returns the magic network of id 0."
        return self._networks[0]
    
    def __len__(self) -> int:
        """
        Number of declared networks. 

        Called if len() is applied on a NetworkManager class.
        """
        return len(self._networks)
    
    def __repr__(self):
        """
        Instead of showing the object pointer, we present the information of
        registered networks.
        """
        string = "<{}.{} object at {}>\n".format( self.__class__.__module__, self.__class__.__name__, hex(id(self)))

        string += "Number of registered networks = " + str(len(self)) + "\n"

        for net_id in range(len(self._networks)):
            string += "Network " + str(net_id) + (" (MagicNetwork)" if net_id == 0 else " ") + "\n"
            string += "  populations = ["
            for pop in self._networks[net_id].get_populations():
                string += pop.__class__.__name__ + " at " + hex(id(pop)) + ", "
            string += "]\n"

            string += "  projections = ["
            for proj in self._networks[net_id].get_projections():
                string += proj.__class__.__name__ + " at " + hex(id(proj)) + ", "
            string += "]\n"

            string += "  monitors = ["
            for mon in self._networks[net_id].get_monitors():
                string += mon.__class__.__name__ + " at " + hex(id(mon)) + ", "
            string += "]\n"

            string += "  extensions = ["
            for mon in self._networks[net_id].get_extensions():
                string += mon.__class__.__name__ + " at " + hex(id(mon)) + ", "
            string += "]\n"

            string += "  cyInstance = " + str(self._networks[net_id].instance) + " at " + hex(id(self._networks[net_id].instance)) + "\n"

        return string
        

    def remove_network(self, py_instance):
        """
        Remove the given network from the list of compilable/instantiable networks.
        It is important to invalidate only the slot. If del is called on the dictionary entry then this will lead to a removal of the space and therefore all subsequent networks would be assigned wrong.

        This function will be called from the Network.__del__() method, after destruction of the attached objects.
        """
        Messages._debug("Remove network", py_instance)

        net_id = -1
        for net_id, inst in enumerate(self._networks):
            if inst == py_instance:
                self._networks[net_id] = None

    def clear(self):
        """
        Remove all assigned networks and restore the initial state.
        """
        # destroy the magic network. The other networks are
        # destroyed through the Network.__del__()
        for pop in self._networks[0]._data.populations:
            pop._clear()

        for proj in self._networks[0]._data.projections:
            proj._clear()

        for mon in self._networks[0]._data.monitors:
            mon._clear()

        # In some cases, we dont want to remove
        disable_rm_directory = ConfigManagement.get_global_config('debug') or ConfigManagement.get_global_config('disable_shared_library_time_offset')
        if disable_rm_directory:
            pass

        # Check whether the magic network has been compiled
        elif self._networks[0].compiled:

            network_directory = self._networks[0].directory

            # Removes the library used in last running instance
            if os.path.isfile(network_directory+'/ANNarchyCore0.so'):
                os.remove(network_directory+'/ANNarchyCore0.so')
            if os.path.isfile(network_directory+'/ANNarchyCore0.dylib'):
                os.remove(network_directory+'/ANNarchyCore0.dylib')

            try:
                if os.path.isdir(network_directory):
                    os.rmdir(network_directory)

            except OSError as err:
                # we notice a not empty directory error
                if ConfigManagement.get_global_config('debug') or ConfigManagement.get_global_config('verbose'):
                    Messages._warning("Attempted to clear:", network_directory, "using os.rmdir failed ... retry with shutil")

                # wait a bit so that the OS has time to finish deleting the content of the directory.
                # re-try it with shutil, if it again fails, we continue, the folder is empty ...
                time.sleep(5)
                shutil.rmtree(network_directory, ignore_errors=True)

            self._networks[0]._data.directory = None

        # This will trigger as last consequence
        # Network.__del__()
        #del self._networks
        self._initialize()

    ################################
    ## Memory management
    ################################
    def _cpp_memory_footprint(self, net_id):
        """
        Print the C++ memory consumption for populations, projections on the console.

        :param net_id: net_id of the requested network.
        """
        from ANNarchy.core.Global import _bytes_human_readable

        print("Memory consumption of C++ objects (Network {id}): ".format(id=net_id))

        for pop in self._networks[net_id]._data.populations:
            print(pop.name, _bytes_human_readable(pop._size_in_bytes()))

        for proj in self._networks[net_id]._data.projections:
            print(proj.name, _bytes_human_readable(proj._size_in_bytes()))

        for mon in self._networks[net_id]._data.monitors:
            print(mon.name, _bytes_human_readable(mon._size_in_bytes()))

 