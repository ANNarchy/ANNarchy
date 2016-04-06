***********************************
Simulation
***********************************

Compiling the network
=====================

Once all the relevant information has been defined, one needs to actually compile the network, by calling the ``ANNarchy.compile()`` method:

.. code-block:: python

    compile()
    
The optimized C++ code will be generated in the ``annarchy/`` subfolder relative to your script, compiled, the underlying objects created and made available to the Python interface.

You can specify the following arguments to ``compile()``:

* ``directory``: relative path to the directory where files will be generated and compiled (default: ``annarchy/``)
* ``populations`` and ``projections``: to compile only a subpart of the network, see :doc:`Network`.
* ``compiler``: to select which C++ compiler will be used. By default ``g++`` on Linux and ``clang++`` on OS X are used, you can change it here. Note that only these two compilers are supported for now, and that they must be in your ``$PATH``.
* ``compiler_flags``: to select which flags are passed to the compiler. By default it is ``-march=native -O2``, but you can fine-tune it here. Beware that ``-O3`` is most often a bad idea!

Simulating the network
======================

After the network is correctly compiled, the simulation can be run for the specified duration (in milliseconds) through the ``ANNarchy.simulate()`` method:

.. code-block:: python

    simulate(1000.0) # Simulate for 1 second

The provided duration should be a multiple of ``dt``. If not, the number of simulation steps performed will be approximated.

In some cases, you may want to perform only one step of the simulation, instead of specifing the duration. The ``ANNarchy.step()`` can then be used.

.. code-block:: python

    step() # Simulate for 1 step

Early-stopping
==============

In some cases, it is desired to stop the simulation whenever a criterion is fulfilled (for example, a neural integrator exceeds a certain threshold), not after a fixed amount of time.

There is the possibility to define a ``stop_condition`` at the ``Population`` level::

    pop1 = Population( ... , stop_condition = "r > 1.0")

When calling the ``simulate_until()`` method instead of ``simulate()``::

    t = simulate_until(max_duration=1000.0, populations=pop1)

the simulation will be stopped whenever the ``stop_condition`` of ``pop1`` is met, i.e. when the firing rate of *any* neuron of pop1 is above 1.0. If the condition is never met, the simulation will last maximally ``max_duration``. The methods returns the effective duration of the simulation (to compute reaction times, for example).

The ``stop_condition`` can use any logical operation on the parameters and variables of the neuron associated to the population::

    pop1 = Population( ... , stop_condition = "(r > 1.0) and (mp < 2.0)")

By default, the simulation stops when at least one neuron in the population fulfills the criterion. If you want to stop the simulation when *all* neurons fulfill the condition, you can use the flag ``all`` after the condition::

    pop1 = Population( ... , stop_condition = "r > 1.0 : all")

The flag ``any`` is the default behavior and can be omitted.

The stop criterion can depend on several populations, by providing a list of populations to the ``populations`` argument instead of a single population::

    t = simulate_until(max_duration=1000.0, populations=[pop1, pop2])

The simulation will then stop when the criterion is met in both populations at the same time. If you want that the simulation stops when at least one population meets its criterion, you can specify the ``operator`` argument::

    t = simulate_until(max_duration=1000.0, populations=[pop1, pop2], operator='or')

The default value of ``operator`` is a ``'and'`` function between the populations' criteria.

    
.. warning::

    Global operations (min, max, mean) are not possible inside the ``stop_condition``. If you need them, store them in a variable in the ``equations`` argument of the neuron and use it as the condition::

        equations = """
            r = ...
            max_r = max(r)
        """

Setting inputs periodically
===========================

In most cases, your simulation will be decomposed into a series of fixed-duration trials, where you basically set inputs at the beginning of the trial, run the simulation for a fixed duration, and possibly read out results at the end:

.. code-block:: python

    # Iterate over 100 trials
    result = []
    for trial in range(100):
        # Set inputs to the network
        pop.I = Uniform(0.0, 1.0)
        # Simulate for 1 second
        simulate(1000.)
        # Save the output
        result.append(pop.r)

For convenience, we provide the decorator ``every``, which allows to register a python method and call it automatically during the simulation with a fixed period:

.. code-block:: python

    result = []

    @every(period=1000.)
    def set inputs(n):
        # Set inputs to the network
        pop.I = Uniform(0.0, 1.0)
        # Save the output of the previous step
        if n > 0:
            result.append(pop.r)

    simulate(100 * 1000.)

In this example, ``set_inputs()`` will be executed just before the steps corresponding to times t = 0., 1000., 2000., and so on until t = 100000. 

The method can have any name, but must accept only one argument, the integer ``n`` which will be incremented at each call of the method (i.e. it will take the values 0, 1, 2 until 99). This can for example be used to access data in a numpy array:

.. code-block:: python

    images = np.random.random((100, 640, 480))

    @every(period=1000.)
    def set inputs(n):
        # Set inputs to the network
        pop.I = images[n, :, :]

    simulate(100 * 1000.)

One can define several methods that will be called in the order of their definition:

.. code-block:: python

    @every(period=1000.)
    def set inputs(n):
        pop.I = 1.0

    @every(period=1000.)
    def reset inputs(n):
        pop.I = 0.0

In this example, ``set_inputs()`` will be called first, followed by ``reset_inputs``, so ``pop.I`` will finally be 0.0. The decorator ``every`` accepts an argument ``offset`` defining a delay within the period to call the method:

.. code-block:: python

    @every(period=1000.)
    def set inputs(n):
        pop.I = 1.0

    @every(period=1000., offset=500.)
    def reset inputs(n):
        pop.I = 0.0

In this case, ``set_inputs()`` will be called at times 0, 1000, 2000... while ``reset_inputs()`` will be called at times 500, 1500, 2500..., allowing to structure a trial more effectively. The ``offset`` can be set negative, in which case it will be relative to the end of the trial: 

.. code-block:: python

    @every(period=1000., offset=-100.)
    def reset inputs(n):
        pop.I = 0.0

In this example, the method will be called at times 900, 1900, 2900 and so on. The ``offset`` value can not be longer than the ``period``, by definition. If you try to do so, a modulo operation will anyway be applied (i.e. an offset of 1500 with a period of 1000 becomes 500).

Finally, the ``wait`` argument allows to delay the first call to the method from a fixed interval:

.. code-block:: python

    @every(period=1000., wait=5000.)
    def reset inputs(n):
        pop.I = 0.0

In this case, the method will be called at times 5000, 6000 and so on.

Between two calls to ``simulate()``, the callbacks can be disabled or re-enabled using the following methods:

.. code-block:: python

    @every(period=1000.)
    def reset inputs(n):
        pop.I = 0.0

    # Simulate with callbacks
    simulate(10000.)

    # Disable callbacks
    disable_callbacks()

    # Simulate without callbacks
    simulate(10000.)

    # Re-enable callbacks
    enable_callbacks()

    # Simulate with callbacks
    simulate(10000.)

Note that the period is always relative to the time when ``simulate()`` is called, so if no offset is defined, the callbacks will be called before the first step of a simulation, no matter how long the previous simulation lasted. In the current state, it is not possible yet to enable/disable callbacks selectively, it is all or none.
