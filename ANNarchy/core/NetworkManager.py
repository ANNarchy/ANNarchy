#===============================================================================
#
#     NetworkManager.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Julien Vitay <julien.vitay@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core import Global
import os

class NetworkManager(object):
    """
    This class implements the management of the data for the different networks. The
    add/remove methods allow the 'random' removal of the network instances and later
    refill of empty spaces.

    The class will be placed and instantiated in the ANNarchy.core.Global file.
    """
    def __init__(self):
        """
        Constructor.
        """
        self._create_initial_state()

    def _create_initial_state(self):
        """
        Initialize the container for the initial network.

        Called either from __init__ or clear(). The first
        slot is reserved for the magic network.
        """
        self._network = [
            {
                'populations': [],
                'projections': [],
                'monitors': [],
                'instance': None,
                'compiled': False,
                'directory': os.getcwd() + "/annarchy/"
            },
        ]
        self._py_instances = [None]

    def __getitem__(self, net_id):
        """
        In the ANNarchy framework we need to access easily the network
        configuration. This method makes the NetworkManager easily subscribtable:

            Global._network[net_id]
        """
        if isinstance(net_id, int):
            if net_id < len(self._network):
                return self._network[net_id]
            else:
                Global._error("Network", net_id, "not existing ...")
        elif isinstance(net_id, slice):
            return self._network[net_id]

    def __len__(self):
        """
        Called if len() is applied on a NetworkManager class.
        """
        return len(self._network)

    def add_network(self, py_instance):
        """
        Adds an empty structure for a new network and returns the new network ID.
        """
        new_dict = {
            'populations': [],
            'projections': [],
            'monitors': [],
            'instance': None,
            'compiled': False,
            'directory': os.getcwd() + "/annarchy/"
        }

        found = -1
        # scan for slots which were freed before
        for i, entry in enumerate(self._network):
            if entry == None and self._py_instance[i] == None:
                found = i
                break

        # dependent on the scan append at the end
        # or fill free slot
        if found == -1:
            new_id = len(self._network)
            self._network.append(new_dict)
            self._py_instances.append(py_instance)
        else:
            new_id = found
            self._network[new_id] = new_dict
            self._py_instances[new_id] = py_instance
        Global._debug("Added network", new_id)
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
        Global._debug("Remove network", py_instance)

        net_id = -1
        for net_id, inst in enumerate(self._py_instances):
             if inst == py_instance:
                 self._network[net_id] = None
                 self._py_instances[net_id] = None

    def clear(self):
        """
        Remove all assigned networks and restore the initial state.
        """
        del self._network
        self._create_initial_state()

