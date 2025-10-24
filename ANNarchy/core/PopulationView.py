"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.core.Random import RandomDistribution
from ANNarchy.intern import Messages

class PopulationView :
    """ Container representing a subset of neurons of a Population."""

    def __init__(self, population, ranks, geometry=None):
        """
        Create a view of a subset of neurons within the same population.

        :param population: population object
        :param ranks: list or numpy array containing the ranks of the selected neurons.
        :param geometry: a geometry for the Populationview (optional)
        """
        self.population = population
        "Original (full) population."

        self.ranks = np.array(ranks)
        "Array of ranks in the PopulationView."

        self.geometry = geometry
        "Geometry of the PopulationView (optional)."

        self.size = len(self.ranks)
        "Size of the PopulationView."
        
        # Internal attributes        
        self.neuron_type = self.population.neuron_type
        self.id = self.population.id
        self.offsets = [np.amin(self.ranks), np.amax(self.ranks)+1]
        self.cyInstance = population.cyInstance

    def _copy(self):
        "Returns a copy of the population when creating networks. Internal use only."
        return PopulationView(population=self.population, ranks=self.ranks, geometry=self.geometry)

    ################################
    # Indexing
    ################################

    def __len__(self):
        """
        Number of neurons in the population view.
        """
        return self.size

    def rank_from_coordinates(self, coord, local=False):
        """
        Returns the rank of a neuron based on coordinates.

        When local is False (default), the coordinates are relative to the ORIGINAL population, not the PopulationView.

        When local is True, the coordinates are interpreted relative to the geometry of the PopulationView if available. When you add two population views, the geometry is lost and the method will return an error.

        The rank is relative to the original population. Iterate over len(pop) otherwise.

        :param coord: coordinate tuple, can be multidimensional.
        :param local: whther the coordinates are local to the PopulationView or not (default: False).
        """
        if not local:
            rk = self.population.rank_from_coordinates(coord)
            if not rk in self.ranks:
                Messages._error("There is no neuron of coordinates", coord, "in the PopulationView.")
            return rk

        else:
            if not self.geometry:
                Messages._error("The population view does not have a geometry, cannot use local coordinates.")
            else:
                try:
                    intern_rank = np.ravel_multi_index(coord, self.geometry)
                except:
                    Messages._error("There is no neuron of coordinates", coord, "in a PopulationView of geometry", self.geometry)
                return self.ranks[intern_rank]

    def coordinates_from_rank(self, rank, local=False):
        """
        Returns the coordinates of a neuron based on its rank.

        When local is False (default), the coordinates are relative to the ORIGINAL population, not the PopulationView.

        When local is True, the coordinates are interpreted relative to the geometry of the PopulationView if available. When you add two population views, the geometry is lost and the method will return an error.

        The rank is relative to the original population. Iterate over len(pop) otherwise.

        :param rank: rank of the neuron in the original population
        :param local: whether the coordinates are local to the PopulationView or not (default: False).
        """
        if not local:
            return self.population.coordinates_from_rank(rank)
        else:
            if not self.geometry:
                Messages._error("The population view does not have a geometry, cannot use local coordinates.")
            else:
                if not rank in self.ranks:
                    Messages._error("There is no neuron of rank", rank, "in the PopulationView.")
                intern_rk = self.ranks.index(rank)
                coord = np.unravel_index(intern_rk, self.geometry)
                return coord


    ################################
    # Targets must match the population, both in read and write
    ################################
    @property
    def targets(self) -> list[str]:
        "List of targets connected to the population."
        return self.population.targets

    @targets.setter
    def targets(self, value):
        self.population.targets.append(value)

    @property
    def name(self) -> str:
        "Returns the name of the original population."
        return self.population.name

    @property
    def max_delay(self) -> str:
        return self.population.max_delay

    @property
    def attributes(self) -> list[str]:
        "Returns a list of attributes of the original population."
        return self.population.attributes

    @property
    def variables(self) -> list[str]:
        "Returns a list of variables of the original population."
        return self.population.variables

    @property
    def parameters(self) -> list[str]:
        "Returns a list of constants of the original population."
        return self.population.parameters

    ################################
    ## Access to attributes
    ################################

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'population':
            return object.__getattribute__(self, name)
        elif name == 'spike':
            all_events = set(self.population.spike)
            own_ranks = set(self.ranks)
            return list(sorted(set.intersection(all_events, own_ranks)))
        elif hasattr(self.population, 'attributes'):
            if name in self.population.attributes:
                return self.get(name)
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name == 'population':
            object.__setattr__(self, name, value)
        elif hasattr(self, 'population'):
            if name in self.population.attributes:
                self.set({name: value})
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def get(self, name):
        """
        Returns current variable/parameter value.

        :param name: name of the parameter/variable.
        """
        if name in self.population.attributes:
            all_val = getattr(self.population, name).reshape(self.population.size)
            return all_val[self.ranks]
        else:
            Messages._error("Population does not have a parameter/variable called " + name + ".")

    def set(self, value:dict) -> None:
        """
        Updates the neurons' variable/parameter values.

        :param value: dictionary of parameters/variables to be updated for the corresponding subset of neurons. It can be a single value or a list/1D array of the same size as the PopulationView.

        .. code-block:: python

            >>> subpop = pop[0:5]
            >>> subpop.set( {'tau' : 20, 'r'= np.random.rand(subpop.size) } )

        .. warning::

            If you modify the value of a global parameter, this will be the case for ALL neurons of the population, not only the subset.
        """
        def _set_single(name, rank, value):
            if not self.population.initialized:
                if not name in self.population.neuron_type.description['local']:
                    Messages._error('can not set the value of a global attribute from a PopulationView.')
                    return

                if isinstance(self.population.init[name], np.ndarray):
                    if len(self.population.geometry) == 1:
                        self.population.init[name][rank] = value
                    else: # Need to access the coordinates
                        coords = self.population.coordinates_from_rank(rank)
                        self.population.init[name][coords] = value
                else:
                    val = self.population.init[name]
                    data = val * np.ones(self.population.size)
                    data[rank] = value
                    self.population.init[name] = data.reshape(self.population.geometry)
            else:
                ctype = self.population._get_attribute_cpp_type(name)
                self.population.cyInstance.set_local_attribute(name, rank, value, ctype)

        for val_key in value.keys():
            if hasattr(self.population, val_key):
                # Check the value
                if isinstance(value[val_key], RandomDistribution): # Make sure it is generated only once
                        value[val_key] = np.array(value[val_key].get_values(self.size))
                if isinstance(value[val_key], np.ndarray): # np.array
                    if value[val_key].ndim >1 or len(value[val_key]) != self.size:
                        Messages._error("You can only provide an array of the same size as the PopulationView", self.size)
                        return None
                    if val_key in self.population.neuron_type.description['global']:
                        Messages._error("Global attributes can only have one value in a population.")
                        return None
                    # Assign the value
                    for idx, rk in enumerate(self.ranks):
                        _set_single(val_key, rk, value[val_key][idx])

                elif isinstance(value[val_key], list): # list
                    if len(value[val_key]) != self.size:
                        Messages._error("You can only provide a list of the same size as the PopulationView", self.size)
                        return None
                    if val_key in self.population.neuron_type.description['global']:
                        Messages._error("Global attributes can only have one value in a population.")
                        return None
                    # Assign the value
                    for idx, rk in enumerate(self.ranks):
                        _set_single(val_key, rk, value[val_key][idx])

                else: # single value
                    for rk in self.ranks:
                        _set_single(val_key, rk, value[val_key])
            else:
                Messages._error("the population has no attribute called ", val_key)
                return None

    ################################
    ## Access to weighted sums
    ################################
    def sum(self, target):
        """
        Returns the array of weighted sums corresponding to the target::

            excitatory = pop.sum('exc')

        For spiking networks, this is equivalent to accessing the conductances directly::

            excitatory = pop.g_exc

        If no incoming projection has the given target, the method returns zeros.

        :param target: the desired projection target.

        **Note:** it is not possible to distinguish the original population when the same target is used.
        """
        return self.population.sum(target)[self.ranks]

    ################################
    ## Composition
    ################################

    def __add__(self, other):
        """Allows to join two PopulationViews if they have the same population."""
        from ANNarchy.core.Neuron import IndividualNeuron
        if other.population == self.population:
            if isinstance(other, IndividualNeuron):
                tmp = list(sorted(set(list(self.ranks) + [other.rank])))
                return PopulationView(self.population, np.array(tmp))
            elif isinstance(other, PopulationView):
                tmp = list(sorted(set(list(self.ranks) + list(other.ranks))))
                return PopulationView(self.population, np.array(tmp))
        else:
            Messages._error("can only add two PopulationViews of the same population.")

    def __repr__(self):
        """Defines the printing behaviour."""
        string ="PopulationView of " + str(self.population.name) + '\n'
        string += '  Ranks: ' +  str(self.ranks)
        string += '\n'
        for rk in self.ranks:
            string += '* ' + str(self.population.neuron(rk)) + '\n'
        return string
