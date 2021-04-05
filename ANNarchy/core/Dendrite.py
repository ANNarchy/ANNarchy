#===============================================================================
#
#     Dendrite.py
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
import ANNarchy.core.Global as Global
from ANNarchy.core import Synapse
from ANNarchy.core.Random import RandomDistribution

import numpy as np

class Dendrite(object):
    """
    A ``Dendrite`` is a sub-group of a ``Projection``, gathering the synapses between the pre-synaptic population and a single post-synaptic neuron.

    It can not be created directly, only through a call to ``Projection.dendrite(rank)``:

    ```python
    dendrite = proj.dendrite(6)
    ```
    """
    def __init__(self, proj, post_rank, idx):

        self.post_rank = post_rank
        self.idx = idx
        self.proj = proj
        self.pre = proj.pre

        self.target = self.proj.target

        self.attributes = self.proj.attributes
        self.parameters = self.proj.parameters
        self.variables = self.proj.variables

    @property
    def size(self):
        """
        Number of synapses.
        """
        if self.proj.cyInstance:
            return self.proj.cyInstance.nb_synapses(self.idx)
        return 0

    @property
    def pre_ranks(self):
        """
        List of ranks of pre-synaptic neurons.
        """
        if self.proj.cyInstance:
            return self.proj.cyInstance.pre_rank(self.idx)
        return []

    def __len__(self):
        # Number of synapses.
        
        return self.size

    @property
    def synapses(self):
        """
        Iteratively returns the synapses corresponding to this dendrite.
        """
        for n in self.pre_ranks:
            yield IndividualSynapse(self, n)

    def synapse(self, pos):
        """
        Returns the synapse coming from the corresponding presynaptic neuron.

        :param pos: can be either the rank or the coordinates of the presynaptic neuron
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.proj.pre.rank_from_coordinates(pos)

        if rank in self.pre_ranks:
            return IndividualSynapse(self, rank)
        else:
            Global._error(" The neuron of rank "+ str(rank) + " has no synapse in this dendrite.")
            return None

    # Iterators
    def __getitem__(self, *args, **kwds):
        # Returns the synapse of the given position in the presynaptic population.
        # If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        
        if len(args) == 1:
            return self.synapse(args[0])
        return self.synapse(args)

    def __iter__(self):
        # Returns iteratively each synapse in the dendrite in ascending pre-synaptic rank order.
        for n in self.pre_ranks:
            yield IndividualSynapse(self, n)

    #########################
    ### Access to attributes
    #########################
    def __getattr__(self, name):
        # Method called when accessing an attribute.
        if name == 'proj':
            return object.__getattribute__(self, name)
        elif hasattr(self, 'proj'):
            if name == 'rank': # TODO: remove 'rank' in a future version
                Global._warning("Dendrite.rank: the attribute is deprecated, use Dendrite.pre_ranks instead.")
                return self.proj.cyInstance.pre_rank(self.idx)
            elif name=='pre_rank':
                return self.proj.cyInstance.pre_rank(self.idx)
            elif name == 'delay':
                if self.proj.uniform_delay == -1:
                    return [d*Global.config['dt'] for d in self.proj.cyInstance.get_dendrite_delay(self.idx)]
                else:
                    return self.proj.max_delay * Global.config['dt']
            elif name in self.proj.attributes:
                return getattr(self.proj.cyInstance, 'get_dendrite_'+name)(self.idx)
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        # Method called when setting an attribute.
        if name == 'proj':
            object.__setattr__(self, 'proj', value)
        elif name == 'attributes':
            object.__setattr__(self, 'attributes', value)
        elif hasattr(self, 'proj'):
            if name in self.proj.attributes:
                if name in self.proj.synapse_type.description['local']:
                    if isinstance(value, (np.ndarray, list)):
                        getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.idx, value)
                    else :
                        getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.idx, value * np.ones(self.size))
                else:
                    getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.idx, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def set(self, value):
        """
        Sets the value of a parameter/variable of all synapses.

        Example:

        ```python
        dendrite.set( 'tau' : 20, 'w'= Uniform(0.0, 1.0) } )
        ```

        :param value: a dictionary containing the parameter/variable names as keys.
        """
        for val_key in value.keys():
            if hasattr(self.proj.cy_instance, val_key):
                # Check the type of the data!!
                if isinstance(value[val_key], RandomDistribution):
                    val = value[val_key].getValues(self.size)
                else:
                    val = value[val_key]
                # Set the value
                getattr(self.proj.cyInstance, 'set_dendrite_'+val_key)(self.idx, val)
            else:
                Global._error("Dendrite has no parameter/variable called", val_key)

    def get(self, name):
        """
        Returns the value of a variable/parameter.

        Example:

        ```python
        dendrite.get('w')
        ```

        :param name: name of the parameter/variable.
        """
        if name == 'rank':
            Global._warning("Dendrite.get('rank'): the attribute is deprecated, use Dendrite.pre_ranks instead.")
            return self.proj.cyInstance.pre_rank(self.idx)
        elif name == 'pre_ranks':
            return self.proj.cyInstance.pre_rank(self.idx)
        elif name in self.attributes:
            return getattr(self.proj.cyInstance, 'get_dendrite_'+name)(self.idx)
        else:
            Global._error("Dendrite has no parameter/variable called", name)


    #########################
    ### Formatting
    #########################
    def receptive_field(self, variable = 'w', fill = 0.0):
        """
        Returns the given variable as a receptive field.

        A Numpy array of the same geometry as the pre-synaptic population is returned. 
        Non-existing synapses are replaced by zeros (or the value ``fill``).

        :param variable: name of the variable (default = 'w')
        :param fill: value to use when a synapse does not exist (default: 0.0).
        """
        values = getattr(self.proj.cyInstance, 'get_dendrite_'+variable)(self.idx)
        pre_ranks = self.proj.cyInstance.pre_rank( self.idx )

        m = fill * np.ones( self.pre.size )
        m[pre_ranks] = values

        return m.reshape(self.pre.geometry)


    #########################
    ### Structural plasticity
    #########################
    def create_synapse(self, rank, w=0.0, delay=0):
        """
        Creates a synapse for this dendrite with the given pre-synaptic neuron.

        :param rank: rank of the pre-synaptic neuron
        :param w: synaptic weight (defalt: 0.0).
        :param delay: synaptic delay (default = dt)
        """
        if not Global.config['structural_plasticity']:
            Global._error('"structural_plasticity" has not been set to True in setup(), can not add the synapse.')
            return

        if self.proj.cyInstance.dendrite_index(self.post_rank, rank) != -1:
            Global._error('the synapse of rank ' + str(rank) + ' already exists.')
            return

        # Set default values for the additional variables
        extra_attributes = {}
        for var in self.proj.synapse_type.description['parameters'] + self.proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in self.proj.synapse_type.description['local']:
                if not isinstance(self.proj.init[var['name']], (int, float, bool)):
                    init = var['init']
                else:
                    init = self.proj.init[var['name']]
                extra_attributes[var['name']] = init

        try:
            self.proj.cyInstance.add_synapse(self.post_rank, rank, w, int(delay/Global.config['dt']), **extra_attributes)
        except Exception as e:
            Global._print(e)

    def prune_synapse(self, rank):
        """
        Removes the synapse with the given pre-synaptic neuron from the dendrite.
        
        :param rank: rank of the pre-synaptic neuron
        """
        if not Global.config['structural_plasticity']:
            Global._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapse.')
            return

        if not rank in self.pre_ranks:
            Global._error('the synapse with the pre-synaptic neuron of rank ' + str(rank) + ' did not already exist.')
            return

        self.proj.cyInstance.remove_synapse(self.post_rank, rank)


class IndividualSynapse(object):

    def __init__(self, dendrite, rank):
        self.dendrite = dendrite
        self.rank = rank
        self.idx = self.dendrite.pre_ranks.index(rank)
        self.attributes = self.dendrite.proj.synapse_type.description['local']

    def __getattr__(self, name):
        " Method called when accessing an attribute."

        if name in ['dendrite', 'attributes', 'rank', 'idx']:
            return object.__getattribute__(self, name)
        if name in self.attributes:
            return getattr(self.dendrite.proj.cyInstance, 'get_synapse_'+name)(self.dendrite.idx, self.idx)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name in ['dendrite', 'attributes', 'rank', 'idx']:
            object.__setattr__(self, name, value)
        elif name in self.attributes:
                getattr(self.dendrite.proj.cyInstance, 'set_synapse_'+name)(self.dendrite.idx, self.idx, value)
        else:
            object.__setattr__(self, name, value)
