**********************************
Accessing parameters and variables
**********************************

The state of every population and projection in the network can be accessed by Python both before and after the ``compile()`` command has been called.

Population attributes
=====================

The value of the parameters and variables of all neurons in a population can be accessed and modified through population attributes.

Let's suppose the following neuron type and population have been defined:

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

    pop = Population(name="Pop1", geometry=(8, 8), neuron=LeakyIntegratorNeuron)

You can list all parameters and variables of a population with:

.. code-block:: python

    >>> pop.parameters
    ['tau', 'baseline']
    >>> pop.variables
    ['rate', 'mp']
    
Reading their value is straightforward:

.. code-block:: python

    >>> pop.tau
    10.0
    >>> pop.rate
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

Population-wise parameters/variables have a single value for the population, while neuron-specific ones return a NumPy array with the same geometry has the population.
            
Setting their value is also simple:

.. code-block:: python

    >>> pop.tau = 20.0
    >>> pop.tau
    20.0
    >>> pop.rate = 1.0
    >>> pop.rate
    array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
    >>> pop.mp = 0.5 * np.ones(pop.geometry)
    array([[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]])
    >>> pop.rate = Uniform(0.0, 1.0)
    array([[ 0.97931939,  0.64865327,  0.29740417,  0.49352664,  0.36511704,
             0.59879869,  0.10835491,  0.38481751],
           [ 0.07664157,  0.77532887,  0.04773084,  0.75395453,  0.56072342,
             0.54139054,  0.28553319,  0.96159595],
           [ 0.01811468,  0.30214921,  0.45321071,  0.56728733,  0.24577655,
             0.32798484,  0.84929103,  0.63025331],
           [ 0.34168482,  0.07411291,  0.6510492 ,  0.89025337,  0.31192464,
             0.59834719,  0.77102494,  0.88537967],
           [ 0.41813573,  0.47395247,  0.46603402,  0.45863676,  0.76628989,
             0.42256749,  0.18527079,  0.8322103 ],
           [ 0.70616692,  0.73210377,  0.05255718,  0.01939817,  0.24659769,
             0.50349528,  0.79201573,  0.19159611],
           [ 0.21246111,  0.93570727,  0.68544108,  0.61158741,  0.17954022,
             0.90084004,  0.41286698,  0.45550662],
           [ 0.14720568,  0.51426136,  0.36225438,  0.06096426,  0.77209455,
             0.07348683,  0.43178591,  0.32451531]])


            
For population-wise attributes, you can only specify a single value (float, int or bool depending on the type of the parameter/variable). For neuron-specific attributes, you can provide either:

    * a single value which will be applied to all neurons of the population.
    
    * a list or a one-dimensional Numpy array of the same length as the number of neurons in the population. This information is provided by ``pop.size``.
    
    * a Numpy array of the same shape as the geometry of the population. This information is provided by ``pop.geometry``.
    
    * a random number generator object (Uniform, Normal...).
    
.. note::

    If you do not want to use the attributes of Python (for example when doing a loop over unknown attributes), you can also use the ``get(name)`` and ``set(values)`` methods of **Population**:
    
    .. code-block:: python
        
        pop.get('tau')
        pop.set({'mp': 1.0, 'rate': Uniform(0.0, 1.0)})
        
It is also possible (albeit slower) to iterate over all neurons and change some variables:

.. code-block:: python
    
    for neuron in pop.neurons: 
        if neuron.rank < 8:
            neuron.rate = 1.0
        else:
            neuron.rate = 0.0

Accessing individual neurons
-----------------------------

There is a purely semantic access to individual neurons of a population. The ``IndividualNeuron`` class wraps population data for a specific neuron. It can be accessed through the ``Population.neuron()`` method using either the rank of the neuron (from 0 to ``pop.size - 1``) or its coordinates in the population's geometry:

.. code-block:: python

    >>> print pop.neuron(2, 2)
    Neuron of the population Pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 0.0

It is also possible to index directly the population, as in a Numpy array:

.. code-block:: python

    >>> print pop[2, 2]
    Neuron of the population Pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 0.0

The individual neurons can be manipulated individually:

.. code-block:: python

    >>> my_neuron = pop[2, 2]
    >>> my_neuron.rate = 1.0
    >>> print my_neuron
    Neuron of the population Pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 1.0

.. warning::

    ``IndividualNeuron`` is only a wrapper for ease of use, the real data is stored in arrays for the whole population, so accessing individual neurons is much slower and should be reserved to specific cases (i.e. only from time to time and for a limited set of neurons).

Accessing groups of neurons
-----------------------------
    
Individual neurons can be grouped into ``PopulationView`` objects, which hold references to different neurons of the same population. One can create population views by "adding" several neurons together:

.. code-block:: python

    >>> popview = pop[2,2] + pop[3,3] + pop[4,4]
    >>> popview
    PopulationView of Pop1
      Ranks: [18, 27, 36]
    * Neuron of the population Pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 0.0

    * Neuron of the population Pop1 with rank 27 (coordinates (3, 3)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 0.0

    * Neuron of the population Pop1 with rank 36 (coordinates (4, 4)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      rate = 0.0
    >>> popview.rate = 1.0
    >>> pop.rate
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
           
One can also use the slice operators to create PopulationViews:

.. code-block:: python

    >>> popview = pop[3, :]
    >>> popview.rate = 1.0
    >>> pop.rate 
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

or:

    >>> popview = pop[2:5, 4]
    >>> popview.rate = 1.0
    >>> pop.rate 
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

.. warning::

    Contrary to PyNN, PopulationView in ANNarchy have two limitations:
    
    * The neurons must be from the same population
    * Populationviews can not be used to create Projections.
    
    These limitations will be overcome in a future release.

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

