"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern import ConfigManagement
from ANNarchy.intern import Messages

import os
import shutil
import time

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

    def get_network_dict(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]
        else:
            Messages._error("Network", net_id, "not existing ...")

    def _get_network_ids(self):
        res = []
        for net in self._py_instances:
            if net is not None:
                res.append(net.id)
        return res

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

            network_directory = self._network_desc[0]['directory']
            # Removes the library used in last running instance
            if os.path.isfile(network_directory+'/ANNarchyCore' + str(0) + '.so'):
                os.remove(network_directory+'/ANNarchyCore' + str(0) + '.so')

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

            self._network_desc[0]['directory'] = None

        # This will trigger as last consequence
        # Network.__del__()
        del self._network_desc
        self._create_initial_state()

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

        for pop in self.get_populations(net_id=net_id):
            print(pop.name, _bytes_human_readable(pop.size_in_bytes()))

        for proj in self.get_projections(net_id=net_id):
            print(proj.name, _bytes_human_readable(proj.size_in_bytes()))

        for mon in NetworkManager().get_monitors(net_id=net_id):
            print(mon.name, _bytes_human_readable(mon.size_in_bytes()))

    ################################
    ## Population objects
    ################################
    def get_population(self, net_id, name):
        if net_id < len(self._network_desc):
            for pop in self._network_desc[net_id]['populations']:
                if pop.name == name:
                    return pop
            return None
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_populations(self, net_id):
        if net_id < len(self._network_desc):
            return self._network_desc[net_id]['populations']
        else:
            Messages._error("Network", net_id, "not existing ...")

    def get_population_names(self, net_id):
        if net_id < len(self._network_desc):
            pop_names = []
            for pop in self._network_desc[net_id]['populations']:
                pop_names.append(pop.name)
            return pop_names
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

    ################################
    ## Projection objects
    ################################
    def get_projections(self, net_id, pre=None, post=None, target=None, suppress_error=False) -> list:
        """
        Return the projections attached to network *net_id*. The returned list can be restricted by the
        arguments *pre*, *post*, or *target*.

        HINT: the *suppress_error* flag should only set to *True* in seldom cases.
        """
        if net_id < len(self._network_desc):
            # None of the arguments is set, so we return all
            if post is None and pre is None and target is None:
                return self._network_desc[net_id]['projections']

            # We need to collect the projections according to the criteria
            res = []

            # The user can provide an object or the name, however, the following code
            # expects the population objects.
            if isinstance(post, str):
                obj = self.get_population(post, net_id)
                if obj is None: # Sanity check
                    Messages._error("The post-synaptic population '{}' was not found".format(post))
                post = obj

            if isinstance(pre, str):
                obj = self.get_population(pre, net_id)
                if obj is None: # Sanity check
                    Messages._error("The pre-synaptic population '{}' was not found".format(pre))
                pre = obj

            # All criterias are used
            if post is not None and pre is not None and target is not None:
                for proj in self._network_desc[net_id]['projections']:
                    if proj.post == post and proj.pre == pre and proj.target == target:
                        res.append(proj)

            # post is the criteria
            elif (post is not None) and (pre is None) and (target is None) :
                for proj in self._network_desc[net_id]['projections']:
                    if proj.post == post:
                        res.append(proj)

            # pre is the criteria
            elif (pre is not None) and (post is None) and (target is None):
                for proj in self._network_desc[net_id]['projections']:
                    if proj.pre == pre:
                        res.append(proj)

            # target is the criteria
            elif (target is not None) and (post is None) and (pre is None):
                for proj in self._network_desc[net_id]['projections']:
                    if proj.target == target:
                        res.append(proj)

            # pre/target is the criteria
            elif (pre is not None) and (target is not None) and (post is None) :
                for proj in self._network_desc[net_id]['projections']:
                    if proj.pre == pre and proj.target == target:
                        res.append(proj)

            # post/target is the criteria
            elif (post is not None) and (target is not None) and (pre is None):
                for proj in self._network_desc[net_id]['projections']:
                    if proj.post == post and proj.target == target:
                        res.append(proj)

            # post/pre is the criteria
            elif (post is not None) and (pre is not None) and (target is None):
                for proj in self._network_desc[net_id]['projections']:
                    if proj.post == post and proj.pre == pre:
                        res.append(proj)

            else:
                # for sanity reasons, should not be reached
                raise NotImplementedError

            if not suppress_error and len(res)==0:
                Messages._error("Could not find projections fitting post={post}, pre={pre}, and target={target}"%{'post':post, 'pre': post, 'target':target})

            return res

        else:
            # The network was not in list, we either throw an Exception or return an empty list.
            if not suppress_error:
                Messages._error("Network", net_id, "not existing ...")
            else:
                return []

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

    ################################
    ## Monitor objects
    ################################
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

    ################################
    ## Extensions
    ################################
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

    ################################
    ## Code generation
    ################################
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
