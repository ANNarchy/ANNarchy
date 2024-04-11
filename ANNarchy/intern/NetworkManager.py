"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern import ConfigManagement
from ANNarchy.intern import Messages

import os
import shutil

class NetworkManager :
    """
    This class implements the management of the data for the different networks. The
    add/remove methods allow the 'random' removal of the network instances and later
    refill of empty spaces.

    The class is implemented as singleton and therefore initialized on first request.

    Individual network are accessed by their id and allow the access to their components.
    """
    _instance = None    # singleton instance

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
            cls._create_initial_state(cls._instance)
        
        return cls._instance

    def _create_initial_state(self):
        """
        Initialize the container for the initial network.

        Called either from __init__ or clear(). The first
        slot is reserved for the magic network.
        """
        self._network_desc = [
            {
                'populations': [],
                'projections': [],
                'monitors': [],
                'extensions': [],
                'instance': None,
                'compiled': False,
                'directory': None
            },
        ]
        self._py_instances = [None]

    def get_populations(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['populations']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def add_population(self, net_id, population):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['populations'].append(population)
        else:
            Messages._error("Network", net_id, "not existing ...")

    def number_populations(self, net_id):
        if net_id < len(self._network_desc):
            return len(self._network_desc[net_id]['populations'])
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_projections(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['projections']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def add_projection(self, net_id, projection):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['projections'].append(projection)
        else:
            Messages._error("Network", net_id, "not existing ...")

    def number_projections(self, net_id):
        if net_id < len(self._network_desc):
            return len(self._network_desc[net_id]['projections'])
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_monitors(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['monitors']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def add_monitor(self, net_id, monitor):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['monitors'].append(monitor)
        else:
            Messages._error("Network", net_id, "not existing ...")

    def number_monitors(self, net_id):
        if net_id < len(self._network_desc):
            return len(self._network_desc[net_id]['monitors'])
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_extensions(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['extensions']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def add_extension(self, net_id, extension):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['extensions'].append(extension)
        else:
            Messages._error("Network", net_id, "not existing ...")

    def number_extensions(self, net_id):
        if net_id < len(self._network_desc):
            return len(self._network_desc[net_id]['extensions'])
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_network_dict(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]
        else:
            Messages._error("Network", net_id, "not existing ...")

    def is_compiled(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['compiled']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def set_compiled(self, net_id):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['compiled'] = True
        else:
            Messages._error("Network", net_id, "not existing ...")

    def cy_instance(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['instance']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_code_directory(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['directory']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def set_code_directory(self, net_id, directory):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['directory'] = directory
        else:
            Messages._error("Network", net_id, "not existing ...")

    def set_cy_instance(self, net_id, instance):
        if net_id < len(self._network_desc):
            self._network_desc[net_id]['instance'] = instance
        else:
            Messages._error("Network", net_id, "not existing ...")

    def _remove_last_item_from_list(self, net_id, list_name):
        if net_id >= len(self._network_desc):
            Messages._error("Network", net_id, "not existing ...")
        
        if list_name not in self._network_desc[net_id].keys():
            Messages._error("Field", list_name, "not existing ...")

        self._network_desc[net_id][list_name].pop(-1)

    def __len__(self):
        """
        Called if len() is applied on a NetworkManager class.
        """
        return len(self._network_desc)

    def __repr__(self):
        """
        Instead of showing the object pointer, we present the information of
        registered networks.
        """
        string = "<{}.{} object at {}>\n".format( self.__class__.__module__, self.__class__.__name__, hex(id(self)))

        string += "Number of registered networks = " + str(len(self)) + "\n"

        for net_id in range(len(self._network_desc)):
            string += "Network " + str(net_id) + (" (MagicNetwork)" if net_id == 0 else " ") + "\n"
            string += "  populations = ["
            for pop in self._network_desc[net_id]['populations']:
                string += pop.__class__.__name__ + " at " + hex(id(pop)) + ", "
            string += "]\n"

            string += "  projections = ["
            for proj in self._network_desc[net_id]['projections']:
                string += proj.__class__.__name__ + " at " + hex(id(proj)) + ", "
            string += "]\n"

            string += "  monitors = ["
            for mon in self._network_desc[net_id]['monitors']:
                string += mon.__class__.__name__ + " at " + hex(id(mon)) + ", "
            string += "]\n"

            string += "  extensions = ["
            for mon in self._network_desc[net_id]['extensions']:
                string += mon.__class__.__name__ + " at " + hex(id(mon)) + ", "
            string += "]\n"

            string += "  cyInstance = " + str(self._network_desc[net_id]['instance']) + "\n"

        return string

    def add_network(self, py_instance):
        """
        Adds an empty structure for a new network and returns the new network ID.
        """
        new_dict = {
            'populations': [],
            'projections': [],
            'monitors': [],
            'extensions': [],
            'instance': None,
            'compiled': False,
            'directory': None
        }

        found = -1
        # scan for slots which were freed before
        for i, entry in enumerate(self._network_desc):
            if entry == None and self._py_instances[i] == None:
                found = i
                break

        # dependent on the scan append at the end
        # or fill free slot
        if found == -1:
            new_id = len(self._network_desc)
            self._network_desc.append(new_dict)
            self._py_instances.append(py_instance)
        else:
            new_id = found
            self._network_desc[new_id] = new_dict
            self._py_instances[new_id] = py_instance

        Messages._debug("Added network", new_id)
        return new_id

    def _remove_network(self, py_instance):
        """
        Remove the given network from the list of compilable/instantiable networks.
        It is important to invalidate only the slot. If del is called on the dictionary
        entry then this will lead to a removal of the space and therefore all succesequent
        networks would be assigned wrong.

        This function will be called from the Network.__del__() method, after destruction
        of the attached objects.
        """
        Messages._debug("Remove network", py_instance)

        net_id = -1
        for net_id, inst in enumerate(self._py_instances):
            if inst == py_instance:
                self._network_desc[net_id] = None
                self._py_instances[net_id] = None

    def clear(self):
        """
        Remove all assigned networks and restore the initial state.
        """
        # destroy the magic network. The other networks are
        # destroyed through the Network.__del__()
        for pop in self._network_desc[0]['populations']:
            pop._clear()

        for proj in self._network_desc[0]['projections']:
            proj._clear()

        for mon in self._network_desc[0]['monitors']:
            mon._clear()

        # In some cases, we dont want to remove
        disable_rm_directory = ConfigManagement.get_global_config('debug') or ConfigManagement.get_global_config('disable_shared_library_time_offset')
        if disable_rm_directory:
            pass

        elif self._network_desc[0]['directory'] != None:
            # Removes the library used in last running instance
            if os.path.isfile(self._network_desc[0]['directory']+'/ANNarchyCore' + str(0) + '.so'):
                os.remove(self._network_desc[0]['directory']+'/ANNarchyCore' + str(0) + '.so')

            try:
                if os.path.isdir(self._network_desc[0]['directory']):
                    os.rmdir(self._network_desc[0]['directory'])

            except OSError as err:
                # we notice a not empty directory error
                if err.errno == 39:
                    if ConfigManagement.get_global_config('debug') or ConfigManagement.get_global_config('verbose'):
                        Messages._warning("Attempted to clear:", self._network_desc[0]['directory'], "using os.rmdir failed ... retry with shutil")

                    # we re-try it with shutil, if it again fails, we ignore it ...
                    shutil.rmtree(self._network_desc[0]['directory'], ignore_errors=True)

                else:
                    # Re-throw other errors
                    raise

            self._network_desc[0]['directory'] = None

        # This will trigger as last consequence
        # Network.__del__()
        del self._network_desc
        self._create_initial_state()

