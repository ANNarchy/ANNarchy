*********************************
Connection patterns
*********************************

Available connector methods
=============================        

For further detailed information about these connectors, please refer to the library reference `Projections <../ANNarchyDoc/Projections.html>`_.
    
connect_all_to_all
-------------------------

*All* neurons of the postsynaptic population form connections with *all* neurons of the presynaptic population (dense connectivity). Self-connections are avoided by default, but the parameter ``allow_self_connections`` can be set to ``True``:

.. code-block:: python

    proj.connect_all_to_all( weights = 1.0, delays=0, allow_self_connections=False ) 
    
The ``weights`` and ``delays`` arguments accept both single float values (all synapses will take this initial value), as well as random objects allowing to randomly select the initial values for different synapses:
  
.. code-block:: python

    proj.connect_all_to_all( weights = Uniform(0.0, 0.5) ) 
    
For all the predefined connection patterns, ``weights`` and ``delays`` can take random distribution objects as value.

connect_one_to_one
------------------------

A neuron of the postsynaptic population forms a connection with only *one* neuron of the presynaptic population, the one having exactly the same rank. The two populations must have the same geometry:

.. code-block:: python

    proj.connect_one_to_one( weights = 1.0 ) 

Below is a graphical representation of the difference between **all_to_all** and **one_to_one**:

.. image:: ../_static/one2all.*
    :align: center
    :width: 70%


connect_gaussian
------------------

A neuron of the postsynaptic population forms a connection with a limited region of the presynaptic population, centered around the neuron with the same normalized position. Weight values are initialized using a Gaussian function, with a maximal value ``amp`` for the neuron of same position and decreasing with distance (standard deviation ``sigma``):

.. math:: 

    w(x, y) = A \cdot \exp(-\frac{1}{2}\frac{(x-x_c)^2+(y-y_c)^2}{\sigma^2})
    
where :math:`(x, y)` is the position of the presynaptic neuron (normalized to :math:`[0, 1]^d`) and :math:`(x_c, y_c)` is the position of the postsynaptic neuron (normalized to :math:`[0, 1]^d`). A = amp, sigma = :math:`\sigma`.

In order to void creating useless synapses, the parameter ``limit`` can be set to restrict the creation of synapses to the cases where the value of the weight would be superior to ``limit*abs(amp)``. Default is 0.01 (1%).

Self-connections are avoided by default (parameter ``allow_self_connections``). 

The two populations must ave the same number of dimensions, but the number of neurons can vary as the positions of each neuron are normalized in :math:`[0, 1]^d`:

.. code-block:: python

    proj.connect_gaussian( amp=1.0, sigma=0.2, limit=0.001) 

connect_dog
----------------

The same as **connect_gaussian**, except weight values are computed using a Difference-of-Gaussians (DoG), usually positive in the center, negative a bit further away and small at long distances. 

.. math:: 

    w(x, y) = A^+ \cdot \exp(-\frac{1}{2}\frac{(x-x_c)^2+(y-y_c)^2}{\sigma_+^2}) -  A^- \cdot \exp(-\frac{1}{2}\frac{(x-x_c)^2+(y-y_c)^2}{\sigma_-^2})


Weights smaller than ``limit * abs(amp_pos - amp_neg)`` are not created and self-connections are avoided by default (parameter ``allow_self_connections``):


.. code-block:: python

    proj.connect_gaussian(amp_pos=1.0, sigma_pos=0.2, amp_neg=0.3, sigma_neg=0.7, limit=0.001) 
    

The following figure shows the example of a neuron of coordinates (10, 10) in the postsynaptic population, which is connected through the **gaussian** (left) and **dog** (right) projections to a population of geometry 30*30. The X and Y axis denote the coordinates of the presynaptic neurons, while the Z axis is the weight value.

.. image:: ../_static/gaussiandog.*
    :align: center
    :width: 100%


connect_fixed_number_pre
-----------------------------

Each neuron in the postsynaptic population receives connections from a fixed number of neurons of the presynaptic population chosen randomly. It may happen that two postsynaptic neurons are connected to the same presynaptic neuron and that some presynaptic neurons are connected to nothing:

.. code-block:: python

    proj.connect_fixed_number_pre(number = 20, weights=1.0) 
    
``weights`` can also take a random object.

connect_fixed_number_post
-----------------------------

Each neuron in the presynaptic population sends a connection to a fixed number of neurons of the postsynaptic population chosen randomly. It may happen that two presynaptic neurons are connected to the same postsynaptic neuron and that some postsynaptic neurons receive no connection at all:

.. code-block:: python

    proj.connect_fixed_number_post(number = 20, weights=1.0) 

The following figure shows the **fixed_number_pre** (left) and **fixed_number_post** projections between two populations of 4 neurons, with ``number=2``. In **fixed_number_pre**, each postsynaptic neuron receives exactly 2 connections, while in **fixed_number_post**, each presynaptic neuron send exactly two connections:

.. image:: ../_static/fixed_number.*
    :align: center
    :width: 70%


connect_fixed_probability
-------------------------------

For each postsynaptic neuron, there is a fixed probability that it forms a connection with a neuron of the presynaptic population. It is basically a **all_to_all** projection, except some synapses are not created, making the projection sparser:  

.. code-block:: python

    proj.connect_fixed_probability(probability = 0.2, weights=1.0) 


User-defined patterns
==================================

This section describes the creation of user-specific connection patterns in ANNarchy, if the available patterns are not enough. A connection pattern is simply implemented as a method returning a dictionary of synapse entries. 

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
--------------------------


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
--------------------

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
