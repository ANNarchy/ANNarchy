*********************************
Defining connection patterns
*********************************

This section describes the creation of user-specific connection patterns in ANNarchy, if the available patterns are not enough. A connection pattern is simply implemented as a method returning a dictionary of synapse entries. 

How to start
==================================

A connector method must take on the first position the reference to the presynaptic population and the reference to the postsynaptic population as 2nd argument.

.. code-block:: python

    probabilistic_pattern(pre, post, <other arguments>)

As an example, we will recreate the probabilistic connector method, building synapses with a given probability. For this new pattern we need a weight value (common for all synapses) and a probability value as additional arguments.

.. code-block:: python

    def probabilistic_pattern(pre, post, weight, probability):

        synapse_dict = {}

        ... pattern code comes here ...

        return synapse_dict

The probabilistic pattern
=====================================

The connector method needs to return a dictionary of synapses with the following structure: the key consists of a tuple (``pre_rank``, ``post_rank``) representing a connection from presynaptic neuron *pre_rank* towards a postsynaptic neuron *post_rank*. Please note, that its necessary to use the ranks of the neuron. If you use 2D or 3D populations you need to transform the coordinates into rank with the ``rank_from_coordinates`` function. The value of the dictionary is a dictionary containing the synapse variables, at least a value ('w') and delay ('d').

.. code-block:: python

    import numpy as np

    synapse_dict = {}

    for post_rank in xrange(post.size):
        for pre_rank in xrange(pre.size):
            if np.random.random() < probability:
                synapse_dict[(pre_rank, post_rank)] = { 'w': weight, 'd': delay }
                
    return synapse_dict

The first *for* - loop is needed to create all dendrites within the projection. As said before, a dendrite is a collection of synapses corresponding to on postsynaptic neuron. The inner *for* - loop creates the single synapses within the dendrite, based on a stochastic process defining whether the synapse is build up or not. The variable probability is an argument provided to the function.

In the end the complete pattern could be implemented like the following:

.. code-block:: python

    import numpy as np
    
    def probabilistic_pattern(pre, post, weight, probability):
    
        synapse_dict = {}

        for post_rank in xrange(post.size): # All postsynaptic neurons
            for pre_rank in xrange(pre.size): # All presynaptic neurons
                if np.random.random() < probability:
                    synapse_dict[(pre_rank, post_rank)] = { 'w': weight, 'd': 0.0 }
                    
        return synapse_dict

Usage of the pattern
=====================================

To use the pattern within a projection you provide the pattern method to the ``connect_with_func`` method of ``Projection``

.. code-block:: python

    proj = Projection(
        pre = In, 
        post = Out, 
        target = 'inh' 
    ).connect_with_func(method=probabilistic_pattern, weight=1.0, probability=0.3)   

either directly after defining the Projection pattern as above, or afterwards:

.. code-block:: python

    proj.connect_with_func(method=probabilistic_pattern, weight=1.0, probability=0.3)   
