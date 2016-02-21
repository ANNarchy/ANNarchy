======================
Projections
======================

Declaring the projections
=========================

Once the populations are created, one can connect them by creating ``Projection`` instances:

.. code-block:: python

    proj = Projection(
        pre = pop1,
        post = "pop2",
        target = "exc",
        synapse = Oja
   )

* ``pre`` is either the name of the pre-synaptic population or the corresponding *Population* object.

* ``post`` is either the name of the post-synaptic population or the corresponding *Population* object.

* ``target`` is the type of the connection.

.. warning::

    The post-synaptic neuron type must use ``sum(exc)`` in the rate-coded case respectively ``g_exc`` in the spiking case, otherwise the projection will be useless.

* ``synapse`` is an optional argument requiring a *Synapse* instance. If the ``synapse`` argument is omitted, the default synapse will be used:

    * the default rate-coded synapse defines ``psp = w * pre.r``,
    * the default spiking synapse defines ``g_target += w``.

Building the projections
===========================

Creating the **Projection** objects only defines the information that two populations are connected. The synapses must be explicitely created by applying a connector method on the **Projection** object.

To this end, ANNarchy already provides a set of predefined connector methods, but the user has also the possibility to define his own (see :doc:`Connector`).

The pattern can be applied either directly at the creation of the Projection:

.. code-block:: python

    proj = Projection(
        pre = pop1,
        post = pop2,
        target = "exc",
        synapse = Oja
    ).connect_all_to_all( weights = 1.0 )

or afterwards:

.. code-block:: python

    proj = Projection(
        pre = pop1,
        post = pop2,
        target = "exc",
        synapse = Oja
    )
    proj.connect_all_to_all( weights = 1.0 )

The connector method must be called before the network is compiled.


Projection attributes
=====================

Let's suppose the ``Oja`` synapse is used to create the Projection ``proj`` (spiking synapses are accessed similarly):

.. code-block:: python

    Oja = Synapse(
        parameters= """
            tau = 5000.0 : post-synaptic
            alpha = 8.0 : post-synaptic
        """,
        equations = """
            tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w
        """
    )



Global attributes
------------------

The global parameters and variables of a projection (i.e. defined with the ``post-synaptic`` flag) can be accessed directly through attributes:

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

Contrary to population attributes, there is one value per post-synaptic neuron for global parameters. You can change these values, either before or after compilation, by providing:

* a single value, which will be the same for all post-synaptic neurons.

* a list of values, with the same size as the number of neurons receiving synapses (for some sparse connectivity patterns, it may not be the same as the size of the population, so no multidimensional array is accepted).

After compilation (and therefore creation of the synapses), you can access how many post-synaptic neurons receive actual synapses with:

.. code-block:: python

    >>> proj.size
    64


The list of ranks of the post-synaptic neurons receiving synapses is obtained with:

.. code-block:: python

    >>> proj.post_ranks
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

Local attributes
-----------------

**At the projection level**

Local attributes can also be accessed globally through attributes. It will return a list of lists containing the synapse-specific values.

The first index represents the post-synaptic neurons. It has the same length as `proj.post_ranks`. Beware that if some post-synaptic neurons do not receive any connection, this index will not correspond to the ranks.

The second index addresses the pre-synaptic neurons. If the connection is sparse, it also is unrelated to the ranks of the pre-synaptic neurons in their populations.

.. warning::

    Modifying these lists of lists is error-prone, so this method should be avoided if possible.


**At the post-synaptic level**

The local parameters and variables of a projection (synapse-specific) should better be accessed through the **Dendrite** object, which gathers for a single post-synaptic neuron all synapses belonging to the projection.

.. warning::

    As projections are only instantiated after the call to ``compile()``, local attributes of a Projection are only available then. Trying to access them before compilation will lead to an error!


Each dendrite stores the parameters and variables of the corresponding synapses as attributes, as populations do for neurons. You can loop over all post-synaptic neurons receiving synapses with the ``dendrites`` iterator:

.. code-block:: python

    for dendrite in proj.dendrites:
        print dendrite.rank
        print dendrite.size
        print dendrite.tau
        print dendrite.alpha
        print dendrite.w

``dendrite.rank`` returns a list of pre-synaptic neuron ranks. ``dendrite.size`` returns the number of synapses for the considered post-synaptic neuron. Global parameters/variables return a single value (``dendrite.tau``) and local ones return a list (``dendrite.w``).

.. note::

    You can even omit the ``.dendrites`` part of the iterator:

    .. code-block:: python

        for dendrite in proj:
            print dendrite.rank
            print dendrite.size
            print dendrite.tau
            print dendrite.alpha
            print dendrite.w

You can also access the dendrites individually, either by specifying the rank of the post-synaptic neuron:

.. code-block:: python

    dendrite = proj.dendrite(13)
    print dendrite.w

or its coordinates:

.. code-block:: python

    dendrite = proj.dendrite(5, 5)
    print dendrite.w

.. warning::

    You should make sure that the dendrite actually exists before accessing it through its rank, because it is otherwise a ``None`` object.

Connecting population views
============================

``Projections`` are usually understood as a connectivity pattern between two populations. Complex connectivity patterns have to specifically designed (see :doc:`Connector`).

In some cases, it can be much simpler to connect subsets of neurons directly, using built-in connector methods. To this end, the ``Projection`` object also accepts ``PopulationView`` objects (:doc:`Populations`) for the ``pre`` and ``post`` arguments.

Let's suppose we want to connect the (8,8) populations ``pop1`` and ``pop2`` in a all-to-all manner, but only for the (4,4) neurons in the center of these populations. The first step is to create the ``PopulationView`` objects using the slice operator:

.. code-block:: python

    pop1_center = pop1[2:7, 2:7]
    pop2_center = pop2[2:7, 2:7]

They can then be simply used to create a projection:

.. code-block:: python

    proj = Projection(
        pre = pop1_center,
        post = pop2_center,
        target = "exc",
        synapse = Oja
    ).connect_all_to_all( weights = 1.0 )

Each neuron of ``pop2_center`` will receive synapses from all neurons of ``pop1_center``, and only them. Neurons of ``pop2`` which are not in ``pop2_center`` will not receive any synapse.

.. warning::

    If you define your own connector method (:doc:`Connector`) and want to use PopulationViews, you'll need to iterate over the ``ranks`` attribute of the ``PopulationView`` object. Full ``Population`` objects do not have a ``ranks`` attribute, it is implicitely ``range(pop.size)``.

Specifying delays in synaptic transmission
==============================================

By default, synaptic transmission is considered to be instantaneous (or more precisely, it takes one simulation step (``dt``) for a newly computed firing rate to be taken into account by post-synaptic neurons).

In order to take longer propagation times into account in the transmission of information between two populations, one has the possibility to define synaptic delays for a projection. All the built-in connector methods take an argument ``delays`` (default=``dt``), which can be a float (in milliseconds) or a random number generator.


.. code-block:: python

    proj.connect_all_to_all( weights = 1.0, delays = 10.0)
    proj.connect_all_to_all( weights = 1.0, delays = Uniform(1.0, 10.0))

If the delay is not a multiple of the simulation time step (``dt = 1.0`` by default), it will be rounded to the closest multiple. The same is true for the values returned by a random number generator.

.. hint::

    Per design, the minimal possible delay is equal to ``dt``: values smaller than ``dt`` will be replaced by ``dt``. Negative values do not make any sense and are ignored.

.. warning::

    Spiking projections do not accept non-uniform delays yet.

Controlling projections
===================================

**Synaptic transmission, update and plasticity**

It is possible to selectively control synaptic transmission and plasticity at the projection level. The boolean flags ``transmission``, ``update`` and ``plasticity`` can be set for that purpose::

    proj.transmission = False
    proj.update = False
    proj.plasticity = False

* If ``transmission`` is ``False``, the projection is totally shut down: it does not transmit any information to the post-synaptic population (the corresponding weighted sums or conductances are constantly 0) and all synaptic variables are frozen to their current value (including the synaptic weights ``w``).

* If ``update`` is ``False``, synaptic transmission occurs normally, but the synaptic variables are not updated. For spiking synapses, this includes traces when they are computed at each step, but not when they are integrated in an event-driven manner (flag ``event-driven``). Beware: continous synaptic transmission as in `NMDA synapses <SpikeSynapse.html#continuous-synaptic-transmission>`_ will not work in this mode, as internal variables are not updated.

* If only ``plasticity`` is ``False``, synaptic transmission and synaptic variable updates occur normally, but changes to the synaptic weight ``w`` are ignored.

**Disabling learning**

Alternatively, one can use the ``enable_learning()`` and ``disable_learning()`` methods of ``Projection``. The effect of ``disable_learning()`` depends on the type of the projection:

* for rate-coded projections, ``disable_learning()`` is equivalent to ``update=False``: no synaptic variables is updated.
* for spiking projections, it is equivalent to ``plasticity=False``: only the weights are blocked.

The reason of this difference is to allow continuous synaptic transmission and computation of traces. Calling ``enable_learning()`` without arguments resumes the default learning behaviour.

**Periodic learning**

``enable_learning()`` also accepts two arguments ``period`` and ``offset``. ``period`` defines the interval in ms between two evaluations of the synaptic variables. This can be useful when learning should only occur once at the end of a trial. It is recommended not to use ODEs in the equations in this case, as they are numerized according to a fixed time step. ``offset`` defines the time inside the period at which the evaluation should occur. By default, it is 0, so the variable updates will occur at the next step, then after ``period`` ms, and so on. Setting it to -1 will shift the update at the end of the period.

Note that spiking synapses using online evaluation will not be affected by these parameters, as they are event-driven.

Multiple targets
=================

For spiking neurons, it may be desirable that a single synapses activates different currents (or conductances) in the post-synaptic neuron. One example are AMPA/NMDA synapses, where a single spike generates a "classical" AMPA current, plus a voltage-gated slower NMDA current. The following conductance-based Izhikevich is an example::

    RSNeuron = Neuron(
        parameters = """
            a = 0.02
            b = 0.2
            c = -65.
            d = 8.
            tau_ampa = 5.
            tau_nmda = 150.
            vrev = 0.0
        """ ,
        equations="""
            I = g_ampa * (vrev - v) + g_nmda * nmda(v, -80.0, 60.0) * (vrev -v)        
            dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I : init=-65., midpoint
            du/dt = a * (b*v - u) : init=-13.
            tau_ampa * dg_ampa/dt = -g_ampa
            tau_nmda * dg_nmda/dt = -g_nmda
        """ , 
        spike = """
            v >= 30.
        """, 
        reset = """
            v = c
            u += d
        """,
        functions = """
            nmda(v, t, s) = ((v-t)/(s))^2 / (1.0 + ((v-t)/(s))^2)
        """
    ) 

However, ``g_ampa`` and ``g_nmda`` collect by default spikes from different projections, so the weights will not be shared between the "ampa" projection and the "nmda" one. It is therefore possible to specify a list of targets when building a projection, meaning that a single pre-synaptic spike will increase both ``g_ampa`` and ``g_nmda`` from the same weight::

    proj = Projection(pop1, pop2, ['ampa', 'nmda'], STDP)