***********************************
Populations
***********************************

Once the ``Neuron`` objects have been defined, the populations can be created.

Let us suppose we have defined the following neural object in a rate-coded context (everything is similar for spiking networks):

.. code-block:: python
    
    LeakyIntegratorNeuron = Neuron(
        parameters = """
            tau = 10.0
            baseline = -0.2
        """,
        equations = """
            tau * dmp/dt  + mp = baseline + sum(exc)
            r = pos(mp)
        """
    )
    
Creating populations
====================

Populations of neurons are created using the ``Population`` class:

.. code-block:: python

    pop1 = Population(geometry=(8, 8), neuron=LeakyIntegratorNeuron)
    pop2 = Population(geometry=(8, 8), neuron=LeakyIntegratorNeuron, name="pop2")

The rate-coded or spiking nature of the ``Neuron`` instance is irrelevant for the ``Population`` object.

It takes different parameters:      
        
* ``geometry`` defines the number of neurons in the population, as well as its spatial structure (1D/2D/3D or more). For example, a two-dimensional population with 15*10 neurons takes the argument ``(15, 10)``, while a one-dimensional array of 100 neurons would take ``(100,)`` or simply ``100``.

* ``neuron`` indicates the neuron type to use for this population (which must have been defined before). It requires a ``Neuron`` instance.

* ``name`` is an unique string for each population in the network. If ``name`` is omitted, an internal name such as ``Population0`` will be given (the number is incremented every time a new population is defined). Although this argument is optional, it is strongly recommended to give an understandable name to each population: if you somehow "lose" the reference to the ``Population`` object in some portion of your code, you can always retrieve it using the ``get_population(name)`` method.

After creation, each population has several attributes defined (corresponding to the parameters and variables of the ``Neuron`` type) and is assigned a fixed size (``pop1.size`` corresponding to the total number of neurons, here 64) and geometry (``pop1.geometry``, here ``(8, 8)``).

Geometry and ranks
==================

Each neuron in the population has therefore a set of **coordinates** (expressed relative to ``pop1.geometry``) and a **rank** (from 0 to ``pop1.size -1``). The reason is that spatial coordinates are useful for visualization, or when defining a distance-dependent connection pattern, but that ANNarchy internally uses flat arrays for performance reasons.

The coordinates use the matrix notation for multi-dimensional arrays, which is also used by Numpy (for a 2D matrix, the first index represents the row, the second the column). You can therefore use safely the ``reshape()`` method of Numpy to switch between coordinates-based and rank-based representations of an array.

To convert the rank of a neuron to its coordinates (and vice-versa), you can use the `ravel_multi_index <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel_multi_index.html>`_ and `unravel_index <http://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html#numpy.unravel_index>`_ methods of Numpy, but they can be quite slow. The ``Population`` class provides two more efficient methods to do this conversion:

* ``coordinates_from_rank`` returns a tuple representing the coordinates of neuron based on its rank (between 0 and ``size -1``, otherwise an error is thrown).

* ``rank_from_coordinates`` returns the rank corresponding to the coordinates.
  
For example, with ``pop1`` having a geometry ``(8, 8)``:

.. code-block:: python
  
    >>> pop1.coordinates_from_rank(15)
    (1, 7)
    >>> pop1.rank_from_coordinates((4, 6))
    38


Population attributes
=====================

The value of the parameters and variables of all neurons in a population can be accessed and modified through population attributes.

With the previously defined populations, you can list all their parameters and variables with:

.. code-block:: python

    >>> pop1.attributes
    ['tau', 'baseline', 'mp', 'r']
    >>> pop1.parameters
    ['tau', 'baseline']
    >>> pop1.variables
    ['r', 'mp']
    
Reading their value is straightforward:

.. code-block:: python

    >>> pop1.tau
    10.0
    >>> pop1.r
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

    >>> pop1.tau = 20.0
    >>> pop1.tau
    20.0
    >>> pop1.r = 1.0
    >>> pop1.r
    array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
    >>> pop1.mp = 0.5 * np.ones(pop.geometry)
    array([[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]])
    >>> pop1.r = Uniform(0.0, 1.0)
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


            
For population-wide attributes, you can only specify a single value (float, int or bool depending on the type of the parameter/variable). For neuron-specific attributes, you can provide either:

    * a single value which will be applied to all neurons of the population.
    
    * a list or a one-dimensional Numpy array of the same length as the number of neurons in the population. This information is provided by ``pop1.size``.
    
    * a Numpy array of the same shape as the geometry of the population. This information is provided by ``pop1.geometry``.
    
    * a random number generator object (Uniform, Normal...).
    
.. note::

    If you do not want to use the attributes of Python (for example when doing a loop over unknown attributes), you can also use the ``get(name)`` and ``set(values)`` methods of **Population**:
    
    .. code-block:: python
        
        pop1.get('tau')
        pop1.set({'mp': 1.0, 'r': Uniform(0.0, 1.0)})
        

Accessing individual neurons
============================

There exists a purely semantic access to individual neurons of a population. The ``IndividualNeuron`` class wraps population data for a specific neuron. It can be accessed through the ``Population.neuron()`` method using either the rank of the neuron (from 0 to ``pop1.size - 1``) or its coordinates in the population's geometry:

.. code-block:: python

    >>> print pop1.neuron(2, 2)
    Neuron of the population pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      r = 0.0


The individual neurons can be manipulated individually:

.. code-block:: python

    >>> my_neuron = pop1.neuron(2, 2)
    >>> my_neuron.rate = 1.0
    >>> print my_neuron
    Neuron of the population pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      r = 1.0

.. warning::

    ``IndividualNeuron`` is only a wrapper for ease of use, the real data is stored in arrays for the whole population, so accessing individual neurons is much slower and should be reserved to specific cases (i.e. only from time to time and for a limited set of neurons).

Accessing groups of neurons
===========================
    
Individual neurons can be grouped into ``PopulationView`` objects, which hold references to different neurons of the same population. One can create population views by "adding" several neurons together:

.. code-block:: python

    >>> popview = pop1.neuron(2,2) + pop1.neuron(3,3) + pop1.neuron(4,4)
    >>> popview
    PopulationView of pop1
      Ranks: [18, 27, 36]
    * Neuron of the population pop1 with rank 18 (coordinates (2, 2)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      r = 0.0

    * Neuron of the population pop1 with rank 27 (coordinates (3, 3)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      r = 0.0

    * Neuron of the population pop1 with rank 36 (coordinates (4, 4)).
    Parameters:
      tau = 10.0
      baseline = -0.2

    Variables:
      mp = 0.0
      r = 0.0
    >>> popview.r = 1.0
    >>> pop1.r
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

    >>> popview = pop1[3, :]
    >>> popview.r = 1.0
    >>> pop1.r 
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

or:

    >>> popview = pop1[2:5, 4]
    >>> popview.r = 1.0
    >>> pop1.r
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

``PopulationView`` objects can be used to create projections.

.. warning::

    Contrary to the equivalent in PyNN, PopulationViews in ANNarchy can only group neurons from the same population.



