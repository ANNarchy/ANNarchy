#===============================================================================
#
#     PopulationView
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
from ANNarchy.core import Global as Global
from .Random import RandomDistribution
import numpy as np

class PopulationView(object):
    """ Container representing a subset of neurons of a Population."""

    def __init__(self, population, ranks, geometry=None):
        """
        Create a view of a subset of neurons within the same population.

        :param population: population object
        :param ranks: list or numpy array containing the ranks of the selected neurons.
        :param geometry: a geometry for the Populationview (optional)
        """
        self.population = population
        self.ranks = ranks
        self.geometry = geometry
        self.size = len(self.ranks)

        # For people using Individual neuron
        if self.size == 1:
            self.rank = self.ranks[0]
        else:
            self.rank = self.ranks

        self.neuron_type = self.population.neuron_type
        self.id = self.population.id
        self.name = population.name
        self.cyInstance = population.cyInstance
        self.variables = population.variables
        self.attributes = population.attributes
        self.max_delay = population.max_delay

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
                Global._error("There is no neuron of coordinates", coord, "in the PopulationView.")
            return rk

        else:
            if not self.geometry:
                Global._error("The population view does not have a geometry, cannot use local coordinates.")
            else:
                try:
                    intern_rank = np.ravel_multi_index(coord, self.geometry)
                except:
                    Global._error("There is no neuron of coordinates", coord, "in a PopulationView of geometry", self.geometry)
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
                Global._error("The population view does not have a geometry, cannot use local coordinates.")
            else:
                if not rank in self.ranks:
                    Global._error("There is no neuron of rank", rank, "in the PopulationView.")
                intern_rk = self.ranks.index(rank)
                coord = np.unravel_index(intern_rk, self.geometry)
                return coord


    ################################
    # Targets must match the population, both in read and write
    ################################
    @property
    def targets(self):
        "List of targets connected to the population."
        return self.population.targets

    @targets.setter
    def targets(self, value):
        self.population.targets.append(value)

    ################################
    ## Access to attributes
    ################################

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'population':
            return object.__getattribute__(self, name)
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
            Global._error("Population does not have a parameter/variable called " + name + ".")

    def set(self, value):
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
                    Global._error('can not set the value of a global attribute from a PopulationView.')
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
                getattr(self.population.cyInstance, 'set_single_'+name)(rank, value)

        for val_key in value.keys():
            if hasattr(self.population, val_key):
                # Check the value
                if isinstance(value[val_key], RandomDistribution): # Make sure it is generated only once
                        value[val_key] = np.array(value[val_key].get_values(self.size))
                if isinstance(value[val_key], np.ndarray): # np.array
                    if value[val_key].ndim >1 or len(value[val_key]) != self.size:
                        Global._error("You can only provide an array of the same size as the PopulationView", self.size)
                        return None
                    if val_key in self.population.neuron_type.description['global']:
                        Global._error("Global attributes can only have one value in a population.")
                        return None
                    # Assign the value
                    for idx, rk in enumerate(self.ranks):
                        _set_single(val_key, rk, value[val_key][idx])

                elif isinstance(value[val_key], list): # list
                    if len(value[val_key]) != self.size:
                        Global._error("You can only provide a list of the same size as the PopulationView", self.size)
                        return None
                    if val_key in self.population.neuron_type.description['global']:
                        Global._error("Global attributes can only have one value in a population.")
                        return None
                    # Assign the value
                    for idx, rk in enumerate(self.ranks):
                        _set_single(val_key, rk, value[val_key][idx])

                else: # single value
                    for rk in self.ranks:
                        _set_single(val_key, rk, value[val_key])
            else:
                Global._error("the population has no attribute called ", val_key)
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
                return PopulationView(self.population, list(set(self.ranks + [other.rank])))
            elif isinstance(other, PopulationView):
                return PopulationView(self.population, list(set(self.ranks + other.ranks)))
        else:
            Global._error("can only add two PopulationViews of the same population.")

    def __repr__(self):
        """Defines the printing behaviour."""
        string ="PopulationView of " + str(self.population.name) + '\n'
        string += '  Ranks: ' +  str(self.ranks)
        string += '\n'
        for rk in self.ranks:
            string += '* ' + str(self.population.neuron(rk)) + '\n'
        return string
