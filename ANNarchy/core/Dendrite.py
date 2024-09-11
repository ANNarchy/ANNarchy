"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages

from typing import Iterator
import numpy as np

class Dendrite :
    """
    A `Dendrite` is a sub-group of a `Projection`, gathering the synapses between the pre-synaptic population and a single post-synaptic neuron.

    It can not be created directly, only through a call to ``Projection.dendrite(rank)``:

    ```python
    dendrite = proj.dendrite(6)
    ```
    """
    def __init__(self, proj, post_rank, idx):

        self.post_rank = post_rank
        "Rank of the post-synaptic neuron."
        self.idx = idx
        self.proj = proj
        "Parent projection."
        self.pre = proj.pre

        self.target = self.proj.target

        self.attributes = self.proj.attributes
        self.parameters = self.proj.parameters
        self.variables = self.proj.variables

    @property
    def size(self) -> int:
        """
        Number of synapses.
        """
        if self.proj.cyInstance:
            return self.proj.cyInstance.dendrite_size(self.idx)
        return 0

    @property
    def pre_ranks(self) -> list[int]:
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
    def synapses(self) -> Iterator["IndividualSynapse"]:
        """
        Iteratively returns the synapses corresponding to this dendrite.
        """
        for n in self.pre_ranks:
            yield IndividualSynapse(self, n)

    def synapse(self, pos:int|tuple[int]) -> "IndividualSynapse":
        """
        Returns the synapse coming from the corresponding presynaptic neuron.

        :param pos: can be either the rank or the coordinates of the presynaptic neuron
        :returns: `IndividualSynapse` wrapper instance.
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.proj.pre.rank_from_coordinates(pos)

        if rank in self.pre_ranks:
            return IndividualSynapse(self, rank)
        else:
            Messages._error(" The neuron of rank "+ str(rank) + " has no synapse in this dendrite.")
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
                Messages._warning("Dendrite.rank: the attribute is deprecated, use Dendrite.pre_ranks instead.")
                return self.proj.cyInstance.pre_rank(self.idx)
            
            elif name == 'pre_rank':
                return self.proj.cyInstance.pre_rank(self.idx)
            
            elif name == 'delay':
                if self.proj.uniform_delay == -1:
                    return [d*get_global_config('dt') for d in self.proj.cyInstance.get_dendrite_delay(self.idx)]
                else:
                    return self.proj.max_delay * get_global_config('dt')
            
            elif name == "w" and self.proj._has_single_weight():
                return self.proj.cyInstance.get_global_attribute(name, get_global_config('precision'))
            
            elif name in self.proj.attributes:
                # Determine C++ data type
                ctype = None
                for var in self.proj.synapse_type.description['variables']+self.proj.synapse_type.description['parameters']:
                    if var['name'] == name:
                        ctype = var['ctype']

                if name in self.proj.synapse_type.description['local']:
                    return self.proj.cyInstance.get_local_attribute_row(name, self.idx, ctype)
                elif name in self.proj.synapse_type.description['semiglobal']:
                    return self.proj.cyInstance.get_semiglobal_attribute(name, self.idx, ctype)
                else:
                    return self.proj.cyInstance.get_global_attribute(name, ctype)
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
                # Determine C++ data type
                ctype = None
                for var in self.proj.synapse_type.description['variables']+self.proj.synapse_type.description['parameters']:
                    if var['name'] == name:
                        ctype = var['ctype']

                if name in self.proj.synapse_type.description['local']:
                    if isinstance(value, (np.ndarray, list)):
                        self.proj.cyInstance.set_local_attribute_row(name, self.idx, value, ctype)
                    else:
                        self.proj.cyInstance.set_local_attribute_row(name, self.idx, value * np.ones(self.size), ctype)

                elif name in self.proj.synapse_type.description['semiglobal']:
                    self.proj.cyInstance.set_semiglobal_attribute(name, self.idx, value, ctype)

                else:
                    # HD: will break the execution of the program
                    Messages._error("Projection attributes marked as *projection* should not be updated through dendrites.")
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def set(self, value:dict) -> None:
        """
        Sets the value of a parameter/variable of all synapses.

        Example:

        ```python
        dendrite.set( { 'tau' : 20, 'w'= Uniform(0.0, 1.0) } )
        ```

        :param value: a dictionary containing the parameter/variable names as keys.
        """
        for key, value in value.items():
            # sanity check and then forward to __setattr__
            if key in self.attributes:
                setattr(self, key, value)
            else:
                Messages._error("Dendrite has no parameter/variable called", key)

    def get(self, name:str) -> float:
        """
        Returns the value of a variable/parameter.

        Example:

        ```python
        dendrite.get('w')
        ```

        :param name: name of the parameter/variable.
        :returns: a single value.
        """
        if name == 'rank':
            Messages._warning("Dendrite.get('rank'): the attribute is deprecated, use Dendrite.pre_ranks instead.")
            return self.proj.cyInstance.pre_rank(self.idx)
        elif name == 'pre_ranks':
            return self.proj.cyInstance.pre_rank(self.idx)
        elif name in self.attributes:
            return getattr(self, name)
        else:
            Messages._error("Dendrite has no parameter/variable called", name)


    #########################
    ### Formatting
    #########################
    def receptive_field(self, variable:str='w', fill:float=0.0) -> np.array:
        """
        Returns the given variable as a receptive field.

        A Numpy array of the same geometry as the pre-synaptic population is returned. 
        Non-existing synapses are replaced by zeros (or the value ``fill``).

        :param variable: name of the variable (default = 'w')
        :param fill: value to use when a synapse does not exist (default: 0.0).
        :returns: an array.
        """
        values = getattr(self.proj.cyInstance, 'get_dendrite_'+variable)(self.idx)
        pre_ranks = self.proj.cyInstance.pre_rank( self.idx )

        m = fill * np.ones( self.pre.size )
        m[pre_ranks] = values

        return m.reshape(self.pre.geometry)


    #########################
    ### Structural plasticity
    #########################
    def create_synapse(self, rank:int, w:float=0.0, delay:float=0) -> None:
        """
        Creates a synapse for this dendrite with the given pre-synaptic neuron.

        :param rank: rank of the pre-synaptic neuron
        :param w: synaptic weight.
        :param delay: synaptic delay.
        """
        if not get_global_config('structural_plasticity'):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not add the synapse.')
            return

        if self.proj.cyInstance.dendrite_index(self.post_rank, rank) != -1:
            Messages._error('the synapse of rank ' + str(rank) + ' already exists.')
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
            self.proj.cyInstance.add_synapse(self.post_rank, rank, w, int(delay/get_global_config('dt')), **extra_attributes)
        except Exception as e:
            Messages._print(e)

    def create_synapses(self, ranks:list[int], weights:list[float]=None, delays:list[float]=None) -> None:
        """
        Creates a synapse for this dendrite with the given pre-synaptic neurons.

        :param ranks: list of ranks of the pre-synaptic neurons.
        :param weights: list of synaptic weights (default: 0.0).
        :param delays: list of synaptic delays (default = dt).
        """
        if not get_global_config('structural_plasticity'):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not add the synapse.')
            return

        # No user-side init
        if weights is None:
            weights = [0.0] * len(ranks)

        if delays is None:
            delays = [0] * len(ranks)

        # Collect other attributes than w/delay
        extra_attribute_names = []
        for var in self.proj.synapse_type.description['parameters'] + self.proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in self.proj.synapse_type.description['local']:
                extra_attribute_names.append[var['name']]

        # Create the synapses
        for rank, w, delay in zip(ranks, weights, delays):
            if self.proj.cyInstance.dendrite_index(self.post_rank, rank) != -1:
                Messages._error('the synapse of rank ' + str(ranks) + ' already exists.')
                return

            # Set default values for the additional variables
            extra_attributes = {}
            for var in extra_attribute_names:
                if not isinstance(self.proj.init[var], (int, float, bool)):
                    init = var['init']
                else:
                    init = self.proj.init[var]
                extra_attributes[var] = init

            try:
                self.proj.cyInstance.add_synapse(self.post_rank, rank, w, int(delay/get_global_config('dt')), **extra_attributes)
            except Exception as e:
                Messages._print(e)

    def prune_synapse(self, rank:int) -> None:
        """
        Removes the synapse with the given pre-synaptic neuron from the dendrite.

        :param rank: rank of the pre-synaptic neuron
        """
        if not get_global_config('structural_plasticity'):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapse.')
            return

        if not rank in self.pre_ranks:
            Messages._error('the synapse with the pre-synaptic neuron of rank ' + str(rank) + ' did not already exist.')
            return

        self.proj.cyInstance.remove_synapse(self.post_rank, rank)

    def prune_synapses(self, ranks:list[int]):
        """
        Removes the synapses which belong to the provided pre-synaptic neurons from the dendrite.

        :param ranks: list of ranks of the pre-synaptic neurons.
        """
        if not get_global_config('structural_plasticity'):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapse.')
            return

        for rank in ranks:
            self.prune_synapse(rank)

class IndividualSynapse :

    def __init__(self, dendrite, rank):
        self.dendrite = dendrite
        self.rank = rank

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name in ['dendrite', 'attributes', 'rank']:
            return object.__getattribute__(self, name)

        elif name in self.dendrite.proj.synapse_type.description['attributes']:
            # Determine C++ data type
            ctype = None
            for var in self.dendrite.proj.synapse_type.description['variables']+self.dendrite.proj.synapse_type.description['parameters']:
                if var['name'] == name:
                    ctype = var['ctype']

            if name in self.dendrite.proj.synapse_type.description['local']:
                return self.dendrite.proj.cyInstance.get_local_attribute(name, self.dendrite.idx, self.rank, ctype)
            elif name in self.dendrite.proj.synapse_type.description['semiglobal']:
                return self.dendrite.proj.cyInstance.get_semiglobal_attribute(name, self.dendrite.idx, ctype)
            else:
                return self.dendrite.proj.cyInstance.get_global_attribute(name, ctype)

        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name in ['dendrite', 'attributes', 'rank']:
            object.__setattr__(self, name, value)

        elif name in self.dendrite.proj.synapse_type.description['attributes']:
            # Determine C++ data type
            ctype = None
            for var in self.dendrite.proj.synapse_type.description['variables']+self.dendrite.proj.synapse_type.description['parameters']:
                if var['name'] == name:
                    ctype = var['ctype']

            if name in self.dendrite.proj.synapse_type.description['local']:
                self.dendrite.proj.cyInstance.set_local_attribute(name, self.dendrite.idx, self.rank, value, ctype)
            elif name in self.dendrite.proj.synapse_type.description['semiglobal']:
                self.dendrite.proj.cyInstance.set_semiglobal_attribute(name, self.dendrite.idx, value, ctype)
            else:
                self.dendrite.proj.cyInstance.set_global_attribute(name, value, ctype)

        else:
            object.__setattr__(self, name, value)
