======================
Projections
======================

Once the populations are created, one can connect them by creating **Projection** instances:

.. code-block:: python

    proj = Projection(
        pre = Pop1, 
        post = "population2", 
        target = "exc",
        synapse = Oja
   )
                         
* ``pre`` is either the name of the presynaptic population or the corresponding *Population* object.

* ``post`` is either the name of the postsynaptic population or the corresponding *Population* object.

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
 

Projection attributes
=====================


Let's suppose the following network is defined:

.. code-block:: python
    
    from ANNarchy import *

    LeakyIntegratorNeuron = RateNeuron(
        parameters= """   
            tau = 10.0 : population
            baseline = -0.2
        """,
        equations = """
            tau * dmp / dt + mp = baseline + sum(exc)
            rate = pos(mp)
        """
    )

    Oja = RateSynapse(
        parameters= """   
            tau = 5000.0 : postsynaptic
            alpha = 8.0 : postsynaptic
        """,
        equations = """
            tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value
        """
    ) 

    pop1 = Population(name="Pop1", geometry=(8, 8), neuron=LeakyIntegratorNeuron)
    pop2 = Population(name="Pop2", geometry=(8, 8), neuron=LeakyIntegratorNeuron)

    proj = Projection(
        pre = pop1,
        post = pop2,
        target = "exc",
        synapse = Oja,
    ).connect_all_to_all(weights=Uniform(0.0, 0.5))
    
    
Global attributes
------------------    

The global parameters and variables of a projection (i.e. defined with the ``postsynaptic`` flag) can be accessed directly through attributes:

.. code-block:: python

    >>> proj.tau
    array([ 5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,
            5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.])
            
Contrary to population attributes, there is one value per postsynaptic neuron for global parameters. You can change these values, either before or after compilation, by providing:

* a single value, which will be the same for all postsynaptic neurons.

* a list of values, with the same size as the number of neurons receiving synapses (for some sparse connectivity patterns, it may not be the same as the size of the population, so no multidimensional array is accepted).

You can obtain a list of the postsynaptic neurons receiving synapses with:

.. code-block:: python

    >>> proj.post_ranks
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

Local attributes
-----------------

The local parameters and variables of a projection (synapse-specific) have to be accessed through the **Dendrite** object, which gathers for a single postsynaptic neuron all synapses belonging to the projection. 

.. warning::

    As projections are only instantiated after the call to ``ANNarchy.compile()``, local attributes of a Projection are only available then. Trying to access them before compilation will lead to an error!
    

Each dendrite stores the parameters and variables of the corresponding synapses as attributes, as populations do for neurons. You can loop over all postsynaptic neurons receiving synapses with the ``dendrites`` iterator:

.. code-block:: python

    for dendrite in proj.dendrites:
        print dendrite.rank
        print dendrite.size
        print dendrite.tau
        print dendrite.alpha
        print dendrite.value
        
``dendrite.rank`` returns a list of presynaptic neuron ranks. ``dendrite.size`` returns the number of synapses for the considered postsynaptic neuron. Global parameters/variables return a single value (``dendrite.tau``) or one-dimensional Numpy arrays (``dendrite.values``).

.. note::

    You can even omit the ``.dendrites`` part of the iterator:
    
    .. code-block:: python

        for dendrite in proj:
            print dendrite.rank
            print dendrite.size
            print dendrite.tau
            print dendrite.alpha
            print dendrite.value
        
You can also access the dendrites individually, either by specifying the rank of the postsynaptic neuron:

.. code-block:: python

    dendrite = proj.dendrite(13)
    print dendrite.value
    
or its coordinates:

.. code-block:: python

    dendrite = proj.dendrite(5, 5)
    print dendrite.value
    
.. warning::

    You should make sure that the dendrite actually exist before accessing it through its rank, because it is otherwise a ``None`` object.        
        

        
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
