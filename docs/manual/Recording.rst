***********************************
Recording 
***********************************

Between two calls to ``simulate()``, all neural and synaptic variables can be accessed through the generated attributes (see :doc:`Populations` and :doc:`Projections`). The evolution of neural or synaptic variables during a simulation can be selectively recorded using ``Monitor`` objects (see :doc:`../API/Monitor`).

The monitor object can be created at any time (before or after ``compile()``) to record any variable of a ``Population``, ``PopulationView`` or ``Dendrite``. It is not possible to record whole projections because it would fill the memory too quickly.

.. note::

    The value of each variable is stored for every simulation step in the RAM. For huge networks and long simulations, this can very rapidly fill up the available memory and lead to cache defaults, thereby degrading strongly the performance. It is the user's responsability to record only the needed variables and to regularly save the values in a file.


Neural variables
================

The ``Monitor`` object takes four arguments: 

* ``obj``: the object to monitor. It can be a population, a population view (a slice of a population or an individual neuron) or a dendrite (the synapses of a projection which reach a single post-synaptic neuron).

* ``variables``: a (list of) variable name(s) which should be recorded. They should be variables of the neuron/synapse model of the corresponding object. You can know which variables are recordable by checking the ``variables`` attribute of the object (``pop.variables`` or ``proj.variables``).

* ``period``: the period in ms at which recordings should be made. By default, recording is done after each simulation step (``dt``), but this may be overkill in long simulations.

* ``start``: boolean value stating if the recordings should start immediately after the creation of the monitor (default), or if it should be started later.

Some examples::

    m = Monitor(pop, 'r') # record r in all neurons of pop
    m = Monitor(pop, ['r', 'v']) # record r and v of all neurons
    m = Monitor(pop[:100], 'r', period=10.0) # record r in the first 100 neurons of pop, every 10 ms
    m = Monitor(pop, 'r', start=False) # record r in all neurons, but do not start recording

Spiking networks additionally allow to record the `spike` events in a population (see later). You also can record conductances (e.g. ``g_exc``) and weighted sums of inputs in rate-coded networks (``sum(exc)``) the same way::

    m = Monitor(pop, ['spike', 'g_exc', 'g_inh'])
    m = Monitor(pop, ['r', 'sum(exc)', 'sum(inh)'])

Starting the recordings
------------------------

If ``start`` is set to ``False``, recordings can be started later by calling the ``start()`` method::

    m = Monitor(pop, 'r', start=False)
    simulate(100.)
    m.start()
    simulate(100.)

In this case, only the last 100 ms of the simulation will be recorded. Otherwise, recording would start immediately after the creation of the object.

Pausing/resuming the recordings
-------------------------------

If you are interested in recording only specific periods of the simulation, you can ause and resume recordings::

    m = Monitor(pop, 'r')
    simulate(100.)
    m.pause()
    simulate(1000.)
    m.resume()
    simulate(100.)

In this example, only the first and last 100 ms of the simulation are recorded.

Retrieving the recordings
--------------------------

The recorded values are obtained through the ``get()`` method. If no argument is passed, a dictionary is returned with one element per recorded variable. If the name of a variable is passed (for example ``get('r')``), the recorded values for this variable are directly returned::

    m = Monitor(pop, ['r', 'v'])
    simulate(100.)
    data = m.get()
    simulate(100.)
    r = m.get('r')
    v = m.get('v')

In the example above, ``data`` is a dictionary with two keys ``'r'`` and ``'v'``, while ``r`` and ``v`` are directly the recorded arrays.

The recorded values are Numpy arrays with two dimensions, the **first** one representing **time**, the **second** one representing the ranks of the recorded neurons.

For example, the time course of the firing rate of the neuron of rank 15 is accessed through::

    data['r'][:, 15]

The firing rates of the whole population after 50 ms of simulation are accessed with::

    data['r'][50, :]


.. note::
    
    Once you call ``get()``, the internal data is erased, so calling it immediately afterwards will return an empty recording data. You need to simulate again in order to retrieve new values.

**Representation of time**

The time indices are in simulation steps (integers), not in real time (ms). If ``dt`` is different from 1.0, this indices must be multiplied by ``dt()`` in order to plot real times::

    setup(dt=0.1)
    # ...
    m = Monitor(pop, 'r')
    simulate(100.)
    r = m.get('r')
    plot(dt()*np.arange(100), r[:, 15])

If recordings used the ``pause()`` and ``resume()`` methods, ``get()`` returns only one array with all recordings concatenated. You can access the steps at which the recording started or paused with the ``times()`` method::

    m = Monitor(pop, 'r')
    simulate(100.)
    m.pause()
    simulate(1000.)
    m.resume()
    simulate(100.)
    r = m.get('r') # A (200, N) Numpy array
    print m.times() # {'start': [0, 1100], 'stop': [100, 1200]}


Special case for spiking neurons
--------------------------------

Any variable defined in the neuron type can be recorded. An exception for spiking neurons is the ``spike`` variable itself, which is never explicitely defined in the neuron type but can be recorded:

.. code-block:: python

    m = Monitor(pop, ['v', 'spike'])

Unlike other variables, the binary value of ``spike`` is not recorded at each time step, which would lead to very sparse matrices, but only the times (in steps, not milliseconds) at which spikes actually occur.

As each neuron fires differently (so each neuron will have recorded spikes of different lengths), ``get()`` in this case does not return a Numpy array, but a dictionary associating to each recorded neuron a list of spike times:

.. code-block:: python

    m = Monitor(pop, ['v', 'spike'])
    simulate(100.0)
    data = m.get('spike')
    print data[0] # [23, 76, 98]

In the example above, the neuron of rank ``0`` has spiked 3 times (at t = 23, 76 and 98 ms if ``dt = 1.0``) during the first 100 ms of the simulation.

**Raster plots**

In order to easily display raster plots, the method ``raster_plot()`` is provided to transform this data into an easily plottable format::

    spike_times, ranks = m.raster_plot(data)
    plot(spike_times, ranks, '.')

``raster_plot()`` returns two Numpy arrays, whose length is the total number of spikes emitted during the simulation. The first array contains the spike times (ín ms) while the second contains the ranks of the neurons who fired. They can be directly used t produce the raster plot with Matplotlib.

An example of the use of ``raster_plot()`` can be seen in the :doc:`../example/Izhikevich` section.

**Mean firing rate**

The mean firing rate in the population can be easily calculated using the length of the arrays returned by ``raster_plot``::

    N = 1000 # number of neurons
    duration = 500. # duration of the simulation
    data = m.get('spike')
    spike_times, ranks = m.raster_plot(data)
    print 'Mean firing rate:', len(spike_times)/float(N)/duration*1000., 'Hz.'

For convenience, this value is returned by the ``mean_fr()`` method, which has access to the number of recorded neurons and the duration of the recordings::

    print 'Mean firing rate:', m.mean_fr(data), 'Hz.'

**Firing rates**

Another useful method is ``smoothed_rate()``. It allows to display the instantaneous firing rate of each neuron based on the ``spike`` recordings::

    rates = m.smoothed_rate(data)
    imshow(rates, aspect='auto')

For each neuron, it returns an array with the instantaneous firing rate during the whole simulation. The instantaneous firing rate is computed by inverting the *inter-spike interval* (ISI) between two consecutive spikes, and assigning it to all simulation steps between the two spikes. 

As this value can be quite fluctuating, a ``smooth`` argument in milliseconds can be passed to ``smoothed_rate()`` to apply a low-pass filter on the firing rates: 

.. code-block:: python

    rates = m.smoothed_rate(data, smooth=200.0)
    imshow(rates, aspect='auto')

A smoothed firing rate for the whole population is also accessible through ``population_rate()``::

    fr = m.population_rate(data, smooth=200.0)

**Histogram**

``histogram()`` allows to count the spikes emitted in the whole population during successive bins of the recording duration::

    histo = m.histogram(data, bins=1.0)
    plot(histo)

``bins`` represents the size of each bin, here 1 ms. By default, the bin size is ``dt``. 


Synaptic variables
===================

Recording of synaptic variables such as weights ``w`` during learning is also possible using the monitor object. However, it can very easily lead to important memory consumption. Let's suppose we have a network composed of two populations of 1000 neurons each, fully connected: each neuron of the second population receives 1000 synapses. This makes a total of 1 million synapses for the projection and, supposing the weights ``w`` use the double floating precision, requires 4 MB of memory. If you record ``w`` during a simulation of 1 second (1000 steps, with ``dt=1.0``), the total added memory consumption would already be around 4GB.

To avoid accidental memory fills, ANNarchy forces the user to define which post-synaptic neuron should be recorded. The corresponding dendrite should be simply passed to the monitor:

.. code-block:: python

    dendrite = proj.dendrite(12) # or simply proj[12]
    m = Monitor(dendrite, 'w')
    simulate(1000.0)
    data = m.get('w')

The ``Monitor`` object has the same ``start()``, ``pause()``, ``resume()`` and ``get()`` methods as for populations. ``get()`` returns also 2D Numpy arrays, the first index being time, the second being the index of the synapse. To know to which pre-synaptic neuron it corresponds, use the ``rank`` attribute of the dendrite::

    dendrite.rank # [0, 3, 12, ..]

.. note::

    If you really need to record all weights of a projection, you can do it with the following code, but do not complain if the simulation becomes slow and the RAM is full...

    .. code-block:: python

        monitors = []
        for dendrite in proj:
            monitors.append(Monitor(dendrite, 'w'))
        simulate(1000.0)
        data = []
        for monitor in monitors:
            data.append(monitor.get('w'))



