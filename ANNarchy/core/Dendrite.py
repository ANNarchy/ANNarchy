"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from ANNarchy.core.Random import RandomDistribution

from typing import Iterator
import numpy as np

class Dendrite :
    """
    Sub-group of a `Projection` for a single post-synaptic neuron.

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
        Number of synapses reaching the post-synaptic neuron.
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
                    return [d*ConfigManager().get('dt', self.proj.net_id) for d in self.proj.cyInstance.get_dendrite_delay(self.idx)]
                else:
                    return self.proj.max_delay * ConfigManager().get('dt', self.proj.net_id)
            
            elif name == "w" and self.proj._has_single_weight():
                return getattr(self.proj.cyInstance, "get_global_attribute_"+ConfigManager().get('precision', self.proj.net_id))(name)
            
            elif name in self.proj.attributes:
                # Determine C++ data type
                ctype = None
                for var in self.proj.synapse_type.description['variables']+self.proj.synapse_type.description['parameters']:
                    if var['name'] == name:
                        ctype = var['ctype']

                if name in self.proj.synapse_type.description['local']:
                    return getattr(self.proj.cyInstance, "get_local_attribute_row_"+ctype)(name, self.idx)
                elif name in self.proj.synapse_type.description['semiglobal']:
                    return getattr(self.proj.cyInstance, "get_semiglobal_attribute_"+ctype)(name, self.idx)
                else:
                    return getattr(self.proj.cyInstance, "get_global_attribute_"+ctype)(name)
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
                        getattr(self.proj.cyInstance, "set_local_attribute_row_"+ctype)(name, self.idx, list(value))
                    else:
                        getattr(self.proj.cyInstance, "set_local_attribute_row_"+ctype)(name, self.idx, [value for _ in range(self.size)])

                elif name in self.proj.synapse_type.description['semiglobal']:
                    getattr(self.proj.cyInstance, "set_semiglobal_attribute_"+ctype)(name, self.idx, value)

                else:
                    # HD: will break the execution of the program
                    Messages._error("Projection attributes marked as *projection* should not be updated through dendrites.")
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def set(self, value:dict) -> None:
        """
        Sets the value of a parameter/variable on all synapses in the dendrite.

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
        Returns the value of a parameter/variable.

        ```python
        dendrite.get('w')
        ```

        :param name: name of the parameter/variable.
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

        A numpy array of the same geometry as the pre-synaptic population is returned. 
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
    def create_synapse(self, rank:int, w:float=0.0, delay:float=0) -> None:
        """
        Creates a single synapse for this dendrite with the given pre-synaptic neuron.

        The configuration key `'structural_plasticity'` must be set to `True` before `compile()` for this method to work. 

        ```python
        net = ann.Network()
        net.config(structural_plasticity=True)
        net.compile()

        try:
            proj.dendrite(10).create_synapse(rank=20, w=0.1, delay=0.0)
        except Exception as e:
            print(e)
        ```

        If the synapse already exists, an error is thrown, so make sure to catch the exception.

        :param rank: rank of the pre-synaptic neuron
        :param w: synaptic weight.
        :param delay: synaptic delay in milliseconds that should be a multiple of *dt*.
        """
        if not ConfigManager().get('structural_plasticity', self.proj.net_id):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not add the synapse.')
            return

        if self.proj.cyInstance.synapse_exists(self.post_rank, rank):
            Messages._error(f'Dendrite.create_synapse(): The synapse of rank {rank} already exists for the dendrite {self.post_rank}.')
            return

        # If not all neurons in the post-synaptic population receive connections
        # the post-rank diverge from the LIL index
        if self.proj.cyInstance.nb_dendrites() < self.proj.post.size:
            post_idx = np.where( np.array(self.proj.cyInstance.post_rank()) == self.post_rank )[0][0]
        else:
            post_idx = self.post_rank

        # Set default values for the additional variables
        extra_attributes = []
        for var in self.proj.synapse_type.description['parameters'] + self.proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in self.proj.synapse_type.description['local']:
                if isinstance(self.proj.init[var['name']], (int, float, bool)):
                    init = var['init']
                elif isinstance(self.proj.init[var], RandomDistribution):
                    init = self.proj.init[var].get_value()
                else:
                    init = self.proj.init[var['name']]
                extra_attributes.append(init)

        try:
            self.proj.cyInstance.add_single_synapse(
                post_idx, 
                rank, 
                w, 
                int(delay/ConfigManager().get('dt', self.proj.net_id)), 
                *extra_attributes
            )
        except Exception as e:
            Messages._print(e)
            Messages._error(f'Dendrite.create_synapse(): Could not add synapse of rank {rank} to the dendrite {self.post_rank}.')

    def create_synapses(self, ranks:list[int], weights:float|list[float]=None, delays: float|list[float]=None) -> None:
        """
        Creates a set of synapses for this dendrite with the given pre-synaptic neurons.

        The configuration key `'structural_plasticity'` must be set to `True` before `compile()` for this method to work. 

        ```python
        net = ann.Network()
        net.config(structural_plasticity=True)
        net.compile()

        try:
            proj.dendrite(10).create_synapses(ranks=[20, 30, 40], weights=0.1, delay=0.0)
        except Exception as e:
            print(e)
        ```

        If the synapses already exist, an error is thrown, so make sure to catch the exception.

        :param ranks: list of ranks for the pre-synaptic neurons.
        :param weights: list of synaptic weights (default: 0.0).
        :param delays: list of synaptic delays (default = dt).
        """
        if not ConfigManager().get('structural_plasticity', self.proj.net_id):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not add the synapses.')
            return

        # If not all neurons in the post-synaptic population receive connections
        # the post-rank diverge from the LIL index
        if self.proj.cyInstance.nb_dendrites() < self.proj.post.size:
            post_idx = np.where(self.proj.cyInstance.post_rank, self.post_rank)
        else:
            post_idx = self.post_rank

        # Process pre-synaptic ranks
        if isinstance(ranks, list):
            ranks = np.array(ranks)

        # Process weights
        if weights is None:
            # No user-defined init
            weights = np.array([0.0] * len(ranks))
        elif isinstance(weights, float):
            weights = np.array([weights] * len(ranks))
        elif isinstance(weights, list):
            # User provided a list
            weights = np.array(weights)

        # Process delays
        if delays is None:
            delays = np.array([0] * len(ranks))
        elif isinstance(delays, float):
            delays = np.array([delays] * len(ranks))
        # convert milliseconds -> steps
        delays = np.array(delays/ConfigManager().get('dt', self.proj.net_id), dtype=np.int32)

        # Sort ranks, weights, and delays by ascending ranks
        sorted_indices = np.argsort(ranks)
        ranks = ranks[sorted_indices]
        weights = weights[sorted_indices]
        delays = delays[sorted_indices]

        # Collect other attributes than w/delay
        extra_attributes = []
        for var in self.proj.synapse_type.description['parameters'] + self.proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and var['name'] in self.proj.synapse_type.description['local']:
                if isinstance(self.proj.init[var['name']], (int, float, bool)):
                    init = [self.proj.init[var['name']]] * len(ranks)
                elif isinstance(self.proj.init[var['name']], RandomDistribution):
                    init = self.proj.init[var['name']].get_list_values(len(ranks))
                else:
                    raise AttributeError
                extra_attributes.append(np.array(init))

        # Update connectivity
        try:
            self.proj.cyInstance.add_multiple_synapses(
                post_idx, 
                ranks, 
                weights, 
                delays, 
                *extra_attributes
            )
        except Exception as e:
            Messages._print(e)

    def prune_synapse(self, rank:int) -> None:
        """
        Removes the synapse with the given pre-synaptic neuron from the dendrite.

        The configuration key `'structural_plasticity'` must be set to `True` before `compile()` for this method to work. 

        ```python
        net = ann.Network()
        net.config(structural_plasticity=True)
        net.compile()

        try:
            proj.dendrite(10).prune_synapse(rank=20)
        except Exception as e:
            print(e)
        ```

        If the synapse does not exist, an error is thrown, so make sure to catch the exception.

        :param rank: rank of the pre-synaptic neuron
        """
        if not ConfigManager().get('structural_plasticity', self.proj.net_id):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapse.')
            return

        if not rank in self.pre_ranks:
            Messages._error('the synapse with the pre-synaptic neuron of rank ' + str(rank) + ' did not exist.')
            return

        self.proj.cyInstance.remove_single_synapse(self.post_rank, rank)

    def prune_synapses(self, ranks:list[int]):
        """
        Removes the synapses which belong to the provided pre-synaptic neurons from the dendrite.

        The configuration key `'structural_plasticity'` must be set to `True` before `compile()` for this method to work. 

        ```python
        net = ann.Network()
        net.config(structural_plasticity=True)
        net.compile()

        try:
            proj.dendrite(10).prune_synapses(ranks=[20, 30, 40])
        except Exception as e:
            print(e)
        ```

        If the synapses do not exist, an error is thrown, so make sure to catch the exception.

        :param ranks: list of ranks of the pre-synaptic neurons.
        """
        if not ConfigManager().get('structural_plasticity', self.proj.net_id):
            Messages._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapses.')
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
                return getattr(self.dendrite.proj.cyInstance, "get_local_attribute_"+ctype)(name, self.dendrite.idx, self.rank)
            elif name in self.dendrite.proj.synapse_type.description['semiglobal']:
                return getattr(self.dendrite.proj.cyInstance, "get_semiglobal_attribute_"+ctype)(name, self.dendrite.idx)
            else:
                return getattr(self.dendrite.proj.cyInstance, "get_global_attribute_"+ctype)(name)

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
                getattr(self.dendrite.proj.cyInstance, "set_local_attribute_"+ctype)(name, self.dendrite.idx, self.rank, value)
            elif name in self.dendrite.proj.synapse_type.description['semiglobal']:
                getattr(self.dendrite.proj.cyInstance, "set_semiglobal_attribute_"+ctype)(name, self.dendrite.idx, value)
            else:
                getattr(self.dendrite.proj.cyInstance, "set_global_attribute_"+ctype)(name, value)

        else:
            object.__setattr__(self, name, value)
