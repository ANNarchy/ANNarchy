***********************************
Creating a Network
***********************************

Once the **Neuron** and **Synapse** objects have been defined, the network can be created.

Let us suppose we have defined the following objects in a rate-coded context (everything is similar for spiking networks):

.. code-block:: python

    from ANNarchy import *
    
    LeakyIntegratorNeuron = Neuron(
        parameters = """
            tau = 10.0
            baseline = -0.2
        """,
        equations = """
            tau * dmp / dt  = baseline - mp + sum(exc)
            rate = pos(mp)
        """
    )
    
    Oja = Synapse(
        parameters="""
            tau = 5000
            dt = 1.0
            alpha = 8.0
        """, 
        equations = """
            tau * dvalue / dt = pre.rate * post.rate - alpha * post.rate^2 * value
        """
    )
    
Creating populations
====================

Populations of neurons are created using the **Population** class:

.. code-block:: python

    Pop1 = Population(geometry=(8, 8), neuron=LeakyIntegratorNeuron)
    Pop2 = Population(geometry=(8, 8), neuron=LeakyIntegratorNeuron, name="population2")
      
        
* ``geometry`` indicates the spatial structure of the population (1D/2D/3D or more). For example, a two-dimensional population with 15*10 neurons takes the argument (15, 10), while a 1D array of 100 neurons would take (100,).

* ``neuron`` indicates the neuron type to use for this population (must have been defined before), requires either a *SpikeNeuron* or *RateNeuron* instance.

* ``name`` could be used as an unique ID string for each population in the network. If ``name`` is omitted, an internal name such as ``Population0`` will be given. Although this argument is optional, it is strongly recommended to give an understandable name to each population: if you somehow "lose" the reference to the **Population** object in some portion of your code, you can always retrieve it using the ``ANNarchy.get_population(population_name)`` method.

After creation, each population has attributes defined (corresponding to the parameters and variables of the Neuron type, see section `Accessing parameters and variables <AccessingVariables.html>`_) and has a fixed size (``Pop1.size`` corresponding to the total number of neurons, here 64) and geometry (``Pop1.geometry``, here ``(8, 8)``).

Defining projections
======================

Once the populations are created, one can connect them by creating **Projection** instances:

.. code-block:: python

    proj = Projection(
        pre = Pop1, 
        post = "population2", 
        target = "exc",
        synapse = Oja
   )
                         
* ``pre`` is either the name of the presynaptic population or a *Population* object.

* ``post`` is either the name of the postsynaptic population or a *Population* object.

* ``target`` is the type of the connection. 

.. warning::

    The postsynaptic neuron type must use the ``sum(exc)`` in the rate-coded case respectively ``g_exc`` in the spiking case, otherwise the projection will be useless.
    
* ``synapse`` is an optional argument requiring a *RateSynapse* or *SpikeSynapse* instance.

Creating the projections
===========================

Creating the **Projection** objects only defines the information that two populations are connected. The synapses must be explicitely created by applying a connector method on the **Projection** object.

To this end, ANNarchy already provides a set of predefined connector methods (see next section). The user has also the possibility to define his own connector methods (see `Defining connection patterns <Connector.html>`_)

The pattern can be applied either directly at the creation of the Projection:

.. code-block:: python

    proj = Projection(
        pre = Pop1, 
        post = "population2", 
        target = "exc",
        synapse = Oja
    ).connect_all_to_all( weights = 1.0 )

or afterwards:

.. code-block:: python

    proj = Projection(
        pre = Pop1, 
        post = "population2", 
        target = "exc",
        synapse = Oja
    )
    proj.connect_all_to_all( weights = 1.0 ) 
        
        
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
        
Specifying delays in synaptic transmission
==============================================

By default, synaptic transmission is considered to be instantaneous (or more precisely, it takes one simulation step (``dt``) for a newly computed firing rate to be taken into account by post-synaptic neurons). 

In order to take longer propagation times into account in the transmission of information between two populations, one has the possibility to define synaptic delays for a projection. All the built-in connector methods take an argument ``delays`` (default=0.0), which can be a int, float or random number generator.


.. code-block:: python

    proj.connect_all_to_all( weights = 1.0, delays = 10) 
    proj.connect_all_to_all( weights = 1.0, delays = 10.0) 
    proj.connect_all_to_all( weights = 1.0, delays = Uniform(1.0, 10.0)) 
     
If an ``int`` is given, it is a multiple of the simulation time step (``dt = 1.0`` by default). If a ``float`` is given, it is treated as milliseconds. If the float is not a multiple of ``dt``, it will be rounded to the closest multiple. The same is true for a random number generator.

.. hint::

    Per design, if ``dt = 1.0``, a delay of 1 ms has the same effect as a delay of 0 ms, i.e. the outputs are only perceived in the next computational step. Only delays superior to ``2 * dt`` have an effect.

.. warning::

    Synaptic delays are currently only enabled for rate-coded networks. Synaptic delays for spiking networks will be possible in a future release.

Compiling the network
=====================

Once all this information has been defined, one needs to actually compile the network, by calling the ``ANNarchy.compile()`` method:

.. code-block:: python

    compile()
    
The optimized C++ code will be generated, compiled, the underlying objects created and made available to the Python interface.

Simulating the network
======================

After the network is correctly compiled, the simulation can be run for the specified duration (in milliseconds) through the ``ANNarchy.simulate()`` method:

.. code-block:: python

    simulate(1000.0) # Simulate for 1 second
