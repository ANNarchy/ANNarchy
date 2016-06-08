***********************************
Networks (in parallel)
***********************************

A typical ANNarchy script represents a single network of populations and projections. Most of the work in computational neuroscience consists in running the same network again and again, varying some free parameters each time, until the fit to the data is publishable.  The ``reset()`` allows to return the network to its state before compilation, but this is particularly tedious to implement.

In order to run different networks using the same script, the ``Network`` object can be used to make copies of existing objects (populations, projections and monitors) and simulate them either sequentially or in parallel.

Let's suppose the following dummy network is defined::

    pop1 = PoissonPopulation(100, rates=10.0)
    pop2 = Population(100, Izhikevich)
    proj = Projection(pop1, pop2, 'exc')
    proj.connect_fixed_probability(weights=5.0, probability=0.2)
    m = Monitor(pop2, 'spike')

    compile()

One would like to compare the firing patterns in ``pop2`` when:

* There is no input to ``pop2``.
* The Poisson input is at 10 Hz.
* The Poisson input is at 20 Hz.


Multiple networks
===================

One can create three ``Network`` objects to implement the three conditions::

    net1 = Network()
    net1.add([pop2, m])
    net1.compile()

The network is created empty, and the population ``pop2`` as well as the attached monitor are added to it through the ``add()`` method. This method takes a list of objects (populations, projections and monitors).

The network has then to be compiled by calling the ``compile()`` method specifically on the network. The network can be simulated independently by calling ``simulate()`` or ``simulate_until()`` on the network.

The basic network, with inputs at 10 Hz, can be simulated directly using the normal methods, or copied into a new network::

    net2 = Network()
    net2.add([pop1, pop2, proj, m])
    net2.compile()

Here, all defined objects are added to the network. It would be easier to pass the ``everything`` argument of the Network constructor as ``True``, which has the same effect. We can use this for the third network::

    net3 = Network(everything=True)
    net3.get(pop1).rates = 20.0
    net3.compile()

Here, the population ``pop1`` of the third network has to be accessed though the ``get()`` method. The data corresponding to ``pop1`` will not be the same as for ``net3.get(pop1)``, only the geometry and neuron models are the same.

Once a network is compiled, it can be simulated (but it does not matter if the other networks are also compiled, including the "original" network)::

    net1.simulate(1000.)
    net2.simulate(1000.)
    net3.simulate(1000.)

Spike recordings have to be accessed per network, through the copies of the monitor ``m``::

    t1, n1 = net1.get(m).raster_plot()
    t2, n2 = net2.get(m).raster_plot()
    t3, n3 = net3.get(m).raster_plot()

One can then plot them separately and be not surprised by the fact that the firing rates in ``pop2`` increase with the ones in ``pop1``...

.. note::

    Networks only work on copies of the corresponding objects at the time they are added to the network. It is no use to modify the ``rates`` parameter of ``pop1`` after the network are created.

    Similarly, it is useless to read variables from the original objects if only the networks are simulated: they would still have their original values.

.. warning::

    If you initialize some variables randomly, for example::

        pop2.v = -60. + 10. * np.random.random(100)

    they will have the same value in all networks, they are not drawn again. You need to perform random initialization on each network::

        net1.get(pop2).v = -60. + 10. * np.random.random(100)
        net2.get(pop2).v = -60. + 10. * np.random.random(100)
        net3.get(pop2).v = -60. + 10. * np.random.random(100)

    On the contrary, connection methods having a random components (e.g. ``connect_fixed_probability()`` or using ``weights=Uniform(0.0, 1.0)``) will be redrawn for each network.

.. warning::

    Global simulation methods (:doc:`../API/ANNarchy`) should not be called directly, even with the ``net_id`` parameter. The ``Network`` class overrides them::

        net.step()
        net.simulate()
        net.simulate_until()
        net.reset()
        net.get_time()
        net.set_time(t)
        net.get_current_step()
        net.set_current_step(t)
        net.set_seed(seed)
        net.enable_learning()
        net.disable_learning()
        net.get_population(name)

Parallel simulations
=====================

With independent networks
--------------------------

The three previous networks will be simulated sequentially per definition. As they are very small, they won't beneficiate much from parallelization with OpenMP or CUDA. A potential way to speed-up the computations is to perform the simulations in parallel, what can be useful on a machine with multiple cores.

One has to define a method for the simulation::

    def simulation(idx, net):
        net.simulate(1000.)

The first argument to this method MUST be an integer corresponding to the index of a network, the second MUST be a network object. Other arguments are allowed (see below)

One can then call the ``parallel_run()`` method and pass it the method, as well as a list of networks to apply this network::

    parallel_run(method=simulation, networks=[net1, net2, net3])

This will apply ``simulation()`` in parallel on the 3 networks, reducing the total computation time. ``idx`` will be 0 for ``net1``, 1 for ``net2`` and so on.

``parallel_run()`` returns a list of the values returned by the passed method. For example, instead of accessing all the monitors after the simulation, one could return directly the raster plots::

    def simulation(idx, net):
        net.simulate(1000.)
        return net.get(m).raster_plot()

    results = parallel_run(method=simulation, networks=[net1, net2, net3])

    t1, n1 = results[0]
    t2, n2 = results[1]
    t3, n3 = results[2]


On the same network
-------------------

In the previous example, only ``net1`` is structurally different from the other networks. The networks have to be compiled independently, which can take a long time for complex networks.

A more common use case manipulates a single network and iterates over the values of some parameters to run the exact same simulation. It is possible to use ``parallel_run()`` for that, by passing a ``number`` argument, instead of ``networks``::

    pop1 = PoissonPopulation(100, rates=10.0)
    pop2 = Population(100, Izhikevich)
    proj = Projection(pop1, pop2, 'exc')
    proj.connect_fixed_probability(weights=5.0, probability=0.2)
    m = Monitor(pop2, 'spike')

    compile()

    def simulation(idx, net):
        net.get(pop1).rates = 10. * idx
        net.simulate(1000.)
        return net.get(m).raster_plot()

    results = parallel_run(method=simulation, number = 3)

    t1, n1 = results[0]
    t2, n2 = results[1]
    t3, n3 = results[2]

The ``simulation()`` is called over three internally-created networks (with ``everything=True``). As ``idx = [0, 1, 2]``, the input rates of each network is ``[0, 10., 20.]``, so this method is functionally equivalent to the previous script, with the assumption that an input rate of 0.0 is the same as having no input at all.

As before, the content of the ``simulation()`` method should only manipulate the network object, not the original objects (``pop1.rate = 10. * idx`` won't have any effect).

.. note::

    You do not have access on the internally-created networks after the simulation (they are in a separate memory space). Return the data you want to analyze or write them to disk.

Passing additional arguments
-----------------------------

The two first obligatory arguments of the simulation callback are ``idx``, the index of the network in the simulation, and ``net``, the network object. You can of course use other names, but these two arguments will be passed.

``idx`` can be used for example to access arrays of parameter values::

    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    def simulation(idx, net):
        net.get(pop1).rates = rates[idx]
        ...

    results = parallel_run(method=simulation, number=len(rates))

Another option is to provide additional arguments to the ``simulation`` callback during the ``parallel_run()`` call::

    def simulation(idx, net, rates):
        net.get(pop1).rates = rates
        ...

    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = parallel_run(method=simulation, number=len(rates), rates=rates)

These additional arguments must be lists of the same size as the number of networks (``number`` or ``len(networks)``). You can use as many additional arguments as you want::

    def simulation(idx, net, a, b, c, d):
        ...
    results = parallel_run(method=simulation, number=10, a=..., b=..., c=..., d=...)

In ``parallel_run()``, the arguments can be passed in any order, but they must be named (e.g. ``, a=list(range(0)),``, not ``, list(range(10)),``).
