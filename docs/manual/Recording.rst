***********************************
Recording 
***********************************

Between two calls to ``simulate()``, all neural and synaptic variables can be accessed through the generated attributes (see :doc:`Populations` and :doc:`Projections`). The evolution of neural or synaptic variables during a simulation can be selectively recorded using either global or local methods.

.. warning::

    The value of each variable is stored for every simulation step in the RAM. For huge networks and long simulations, this can very rapidly fill up the available memory and lead to page defaults, thereby degrading strongly the performance. It is the user's responsability to record only the needed variables and to regularly save the values in a file.

Neural variables
================

Neural variables are the cheapest to record, as their size increases linearly with the network. There are two equivalent ways to record them: one is global (scope ``ANNarchy``), the other local to a population.

Recording neural variables globally
-----------------------------------

To record neural variables in several populations (say ``r`` in ``pop1`` and ``r`` and ``mp`` in ``pop2``), the easiest way is to call the method ``start_record()`` with a dictionary taking the population (or its name) as key and a list of variable names as values:

.. code-block:: python

    start_record(
        {
            pop1 : 'r',
            pop2 : ['mp', 'r']
        }
    )

After the call to this method, the value taken by these variables at each time step of the simulation will be stored internally, until the data is retrieved and flushed. It does not matter how many times you call ``simulate()``, the values are accumulated:

.. code-block:: python

    simulate(500.0)
    simulate(500.0)


.. note::

    **New in 4.3.2** : it is possible to add a ``period`` argument (in ms) to ``start_record()`` to record the variables only once every ``period`` ms. The next recording will be at the beginning of the next call to ``simulate()``. Values between two recordings are ignored. This can reduce a lot the memory usage of recordings.

To retrieve the recorded values and free the corresponding memory, you can call the ``get_record()`` method:

.. code-block:: python

    data = get_record()

When ``get_record()`` is called without arguments, the variables defined in the last call to ``start_record()`` will be retrieved. You can also provide a dictionnary to retrieve only certain variables:

.. code-block:: python

    data = get_record( { pop2 : 'r'} )

.. note::
    
    Once you call ``get_record()``, the internal data is flushed, so calling it immediately afterwards will return an empty recording data. You need to simulate again in order to retrieve new values.

The value returned by ``get_record()`` is a nested dictionary encapsulating, for each population and each variable, the values taken by this variable in all neurons over time. These values are represented as a Numpy array, the first index representing the rank of each neuron in the population, the second representing time (in terms of simulation steps, you should multiply by ``dt=1.0`` by default to obtain milliseconds)

.. code-block:: python

    >>> print data[pop2]['r']
    {'stop': 1000, 'start': 0, 'data': array([[   1.        ,    0.89274342,   1.17316076, ...,    1.        ,
             0.82065237,    1.        ],
           [   1.        ,    0.88429849,   1.13928135, ...,    1.        ,
             0.82075133,    1.        ],
           [   1.        ,    0.88807153,   1.16923477, ...,    1.        ,
             0.83248078,    1.        ],
           ..., 
           [   1.        ,    0.88650493,   1.1513879 , ...,    1.        ,
             0.83375699,    1.        ],
           [   1.        ,    0.88153033,   1.13768265, ...,    1.        ,
             0.81927039,    1.        ],
           [   1.        ,    0.88509407,   1.16253288, ...,    1.        ,
             0.81227855,    1.        ]])}
    >>> print data[pop2]['r']['data'].shape
    (64, 1000)

In addition to the ``'data'`` Numpy array actually storing the values, ``'start'`` and ``'stop'`` allow to retrieve the simulation steps corresponding to the start and stop steps of the recordings.

The ``'data'`` array can be used to directly plot the time course of the variable for all neurons:

.. code-block:: python

    from pylab import *
    imshow(data[pop2]['r']['data'], aspect='auto')

or for a single neuron:

.. code-block:: python

    from pylab import *
    plot(data[pop2]['r']['data'][15, :])

.. note::

    By default, ``get_record()`` indexes the neurons of a population by their rank. If you want to manipulate coordinates instead of ranks, you can pass the ``reshape=True`` argument to ``get_record()``:

    .. code-block:: python
    
        >>> data = get_record(reshape=True)
        >>> print data[pop2]['r']['data'].shape
        (8, 8, 1000)

    The first indexes correspond to the population's geometry, the last one to time.

Special case for spiking neurons
--------------------------------

Any variable defined in the neuron type can be recorded using this method. An exception for spiking neurons is the ``spike`` variable itself, which is never explicitely defined in the neuron type but can be recorded:

.. code-block:: python

    start_record(
        {
            pop1 : 'spike',
            pop2 : ['v', 'spike']
        }
    )

Unlike other variables, the binary value of ``spike`` is not recorded at each time step, which would lead to very sparse matrices, but only the times (in steps, not milliseconds) at which spikes actually occur.

As each neuron fires differently (so each neuron will have recorded spikes of different lengths), ``get_record()`` in this case does not return a Numpy array, but a list of lists:

.. code-block:: python

    >>> start_record({ pop1 : 'spike' })
    >>> simulate(100.0)
    >>> data = get_record()
    >>> print data[pop1]['spike']['start']
    0
    >>> print data[pop1]['spike']['stop']
    100
    >>> print len(data[pop1]['spike']['data'])
    64
    >>> print data[pop1]['spike']['data'][0]
    [23, 76, 98]

In the example above, the neuron of rank ``0`` has spiked 3 times (at 23, 76 and 98 ms) during the first 100 ms of simulation (if ``dt = 1.0``).

**Raster plots**

In order to easily display raster plots, the utility function ``raster_plot()`` is provided to transform this data into an easily plottable format:


.. code-block:: python

    spikes = raster_plot(data[pop1]['spike'])
    plot(spikes[:,0], spikes[:,1], '.')

The Numpy array returned by ``raster_plot()`` has two columns and N rows, where N is the total number of spikes emitted by the population during the simulation. The first column represent the time where a spike was emitted, while the second represents the rank of the neuron which fired.

An example of the use of ``raster_plot()`` can be seen in the :doc:`../example/Izhikevich` section.

**Firing rates**

Another utility function is the ``smoothed_rate()`` method. It allows to display the instantaneous firing rate of each neuron based on the ``spike`` recordings.


.. code-block:: python

    rates = smoothed_rate(data[pop1]['spike'])
    imshow(rates, aspect='auto')

For each neuron, it returns an array with the instantaneous firing rate during the whole simulation. The instantaneous firing rate is computed by inverting the *inter-spike interval* (ISI) between two consecutive spikes, and assigning it to all simulation steps between the two spikes. 

As this value can be quite fluctuating, a ``smooth`` argument in milliseconds can be passed to ``smoothed_rate()`` to apply a low-pass filter on the firing rates: 

.. code-block:: python

    rates = smoothed_rate(data[pop1]['spike'], smooth=200.0)
    imshow(rates, aspect='auto')

Stopping the recordings
-----------------------

In some cases, the user may need recordings only in a subpart of the simulation (for example the first and last trials in a learning task). In order to save memory consumption and ease analysis, recording can be temporarily paused or defintely cancelled at any point.

To stop recording:

.. code-block:: python

    >>> start_record({ pop1 : 'r', pop2 : 'r'})
    >>> simulate(1000.0)
    >>> stop_record()
    >>> simulate(10000.0)
    >>> data = get_record()
    >>> print data[pop1]['r']['stop'] - data[pop1]['r']['start']
    1000

After calling ``stop_record()`` you need to call ``start_record()`` again with the same dictionary to allow for further recordings:


.. code-block:: python

    >>> start_record({ pop1 : 'r', pop2 : 'r'})
    >>> simulate(1000.0)
    >>> data_before = get_record()
    >>> stop_record()
    >>> simulate(10000.0)
    >>> start_record({ pop1 : 'r', pop2 : 'r'})
    >>> simulate(1000.0)
    >>> data_after = get_record()
    >>> stop_record()

To avoid passing the dictionary multiple times and storing intermediate values, you can also use the ``pause_record()`` and ``resume_record()`` methods:


.. code-block:: python

    >>> start_record({ pop1 : 'r', pop2 : 'r'})
    >>> simulate(1000.0)
    >>> pause_record()
    >>> simulate(10000.0)
    >>> resume_record()
    >>> simulate(1000.0)
    >>> data = get_record()

In this example, the first and last seconds of the simulation are recorded. The data returned by ``get_record()`` is the concatenation of the two recording sessions. However, the ``start`` and ``stop``  arguments are now lists of times, what allows to find back which part of the matrix belongs to which simulation:

.. code-block:: python

    >>> print data[pop1]['r']['start']
    [0, 11000]
    >>> print data[pop1]['r']['stop']
    [1000, 12000]
    >>> print data[pop1]['r']['data'].shape
    (64, 2000)



Recording neural variables locally
-----------------------------------

For convenience, the methods ``start_record()``, ``get_record()``, ``stop_record``, ``pause_record()`` and ``resume_record()`` are also available for a single population.

* ``start_record()`` only requires a list of variables to record, not a dictionary.
* The dictionary returned by ``get_record()`` starts directly with the recorded variables, not the population.
  
The other methods work as before. This allows a finer control on which populations should be recorded.


.. code-block:: python

    pop1.start_record(['r', 'mp'])
    simulate(1000.0)
    data = pop1.get_record()
    pop1.stop_record()


Synaptic variables
===================

Recording of synaptic variables such as weights ``w`` during learning is also possible. However, it can very easily lead to important memory consumption. Let's suppose we have a network composed of two populations of 1000 neurons each, fully connected: each neuron of the second population receives 1000 synapses. This makes a total of 1 million synapses for the projection and, supposing the weights ``w`` use the double floating precision, requires 4 MB of memory. If you record ``w`` during a simulation of 1 second (1000 steps, with ``dt=1.0``), the total added memory consumption would already be around 4GB.

To avoid accidental memory fills, ANNarchy forces the user to define which postsynaptic neuron should be recorded. Global methods on projections do not work: only methods local to a dendrite (i.e a postsynaptic neuron) do. These methods have the same name and meaning as for populations:

.. code-block:: python

    dendrite = proj.dendrite(12)
    dendrite.start_record(['w'])
    simulate(1000.0)
    data = dendrite.get_record()
    dendrite.stop_record()

.. note::

    If you really need to record all weights of a projection, you can do it with the following code, but do not complain that the simulation becomes slow...

    .. code-block:: python

        for dendrite in proj:
            dendrite.start_record(['w'])
        simulate(1000.0)
        data = []
        for dendrite in proj:
            data.append(dendrite..get_record())
            dendrite.stop_record()    



