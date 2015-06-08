***********************************
Simple STDP
***********************************

The script ``SimpleSTDP.py`` in ``examples/simple_stdp`` shows how to use spike-timing dependent plasticity in a spiking network.

The model is directly adapted from the code provided in the Brian documentation, itself adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001):

    http://brian.readthedocs.org/en/1.4.1/examples-plasticity_STDP1.html

It is simply composed of 1000 input neurons, firing randomly following a Poisson distribution (rate of 15 Hz). They project on a single integrate-and-fire neuron, and the synaptic effiencies are learned using the STDP learning rule.

Defining the neuron
===================

The Poisson neurons are built-in in ANNarchy, so no neuron definition is required here.

The IF neuron used in the model uses linearized excitatory conductances:

.. math::

    \tau_m * \frac{dv}{dt} = (E_l - v) + g_\text{exc} * (Ee - v_r)

where ``v`` is the membrane potential (in mV), :math:`\tau_m` its time constant (10 ms), :math:`E_l` the resting potential (-74 mV), :math:`Ee` the excitatory reversal potential (0 mV) and :math:`v_r` the reset potential (-60 mV).

A spike is emitted whenever the membrane potential exceeds the threshold :math:`vt` (-54 mV), after what it is reset to :math:`vr`.

Synapses are modelled with an exponential decay function: when a pre-synaptic spike arrives, the excitatory conductance :math:`g_\text{exc}` is increased from the value of the corresponding weight, otherwise it progressively decays to 0 using:

.. math::

    \tau_e * \frac{dg_\text{exc}}{dt} + g_\text{exc} = 0

where :math:`\tau_e` is the time constant of the synapse (5 ms).

The implementation of this neuron can then be:

.. code-block:: python

    IF = Neuron(
        parameters = """
            tau_m = 10.0 
            tau_e = 5.0 
            vt = -54.0 
            vr = -60.0 
            El = -74.0 
            Ee = 0.0 
        """,
        equations = """
            tau_m * dv/dt = El - v + g_exc * (Ee - vr) : init = -60.0
            tau_e * dg_exc/dt = -g_exc 
        """,
        spike = """
            v > vt
        """,
        reset = """
            v = vr
        """
    )

The membrane potential ``v`` is initiaized to -60 mV.

Defining the STDP learning rule
===============================

As described in the section :doc:`../manual/SpikeSynapse`, the STDP learning rule can be implemented using its online version. Each synapse needs to update two variables :math:`A_\text{pre}` and :math:`A_\text{post}` which are incremented after each pre- resp. post-synaptic spike and otherwise decay with their own dynamics:

.. math::
    
    \tau_\text{pre} * \frac{dA_\text{pre}}{dt} = - A_\text{pre}

    \tau_\text{post} * \frac{dA_\text{post}}{dt} = - A_\text{post}

When a post-synaptic spike occurs, the synaptic efficiency is increased from :math:`A_\text{pre}` (LTP), while when a pre-synaptic spike occurs, it is decreased from :math:`A_\text{post}`. 

Using the notations of the Brian example and the corresponding parameter values, we can define the following synapse type:

.. code-block:: python

    STDP = Synapse(
        parameters="""
            tau_pre = 20.0 : post-synaptic
            tau_post = 20.0 : post-synaptic
            cApre = 0.01 : post-synaptic
            cApost = -0.0105 : post-synaptic
            wmax = 0.01 : post-synaptic
        """,
        equations = """
            tau_pre * dApre/dt = -Apre : event-driven 
            tau_post * dApost/dt = -Apost : event-driven
        """,
        pre_spike="""
            g_target += w
            Apre += cApre * wmax
            w = clip(w + Apost, 0.0 , wmax)
        """,                  
        post_spike="""
            Apost += cApost * wmax
            w = clip(w + Apre, 0.0 , wmax)
        """
    )

The parameters are flagged with ``post-synaptic`` as they have he same value for all synapses (this reduces considerably the needed memory space). 

**When a pre-synaptic spike occurs:**

* the post-synaptic conductance is increased from ``w``,
* ``Apre`` is incremented,
* the synaptic weight ``w`` is increased from ``Apost``, resulting to LTD as ``Apost`` is negative. We also make sure it stays bounded by 0 and wmax by using the function ``clip()``.
  
**When a post-synaptic spike occurs:**

* ``Apost`` is decremented (as ``cApost`` is negative),
* the synaptic weight is increased from ``Apre``, resulting to LTP.
  
Otherwise, ``Apre`` and ``Apost`` decay to 0 with their own dynamics. This is only simulated, as the integration is performed analytically, using the ``event-driven`` flag.

This online version of STDP is already provided by ANNarchy (:doc:`../API/SpecificSynapse`), so one can simply use:

:: 

    STDP(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.0105, w_max=0.01)

when creating the projections.

.. note::

    The provided STDP synapse uses the PyNN notation for the parameters. In particular ``A_minus`` is positive. The variables ``A_pre`` and ``A_post`` are called ``x`` and ``y``, respectively.

Creating the populations
========================

The first population is composed of spiking neurons firing randomly according to a Poisson distribution with a mean firing rate of 15 Hz. We make use here of the built-in ``PoissonPopulation`` type::

    Input = PoissonPopulation(name = 'Input', geometry=N, rates=F)

with ``F = 15.0`` and ``N = 1000``.

The second population has only one neuron, of the ``IF`` type::

    Output = Population(name = 'Output', geometry=1, neuron=IF)

Connecting the populations
==========================

We first need to create a ``Projection`` with target ``exc`` between the two populations, using the ``STDP`` synapse type::

    proj = Projection( 
        pre = Input, 
        post = Output, 
        target = 'exc',
        synapse = STDP
    )

We then create the synapses and initialize the weights randomly between 0 and ``gmax = 0.01``::

    proj.connect_all_to_all(weights=Uniform(0.0, gmax))

Running the simulation
======================

We must first compile the network::

    compile()

For this simulation, we will record the spiking activity in both populations::

    Mi = Monitor(Input, 'spike') 
    Mo = Monitor(Output, 'spike')   

We can then simulate for 100 seconds (100000 milliseconds)::

    simulate(duration, measure_time=True)

The recorded data is retrieved through ``get_record()``::

    input_spikes = Mi.get('spike')
    output_spikes = Mo.get('spike')

Using the utility function ``smoothed_rate()`` of the monitors (see :doc:`../API/Monitor`) we can compute the mean firing rate of the output neuron, smoothed using a sliding window of 100 ms::

    output_rate = Mo.smoothed_rate(output_spikes, 100.0)

The synaptic weights with the 1000 inputs after learning are simply retrieved with::

    weights = proj.w[0]

Finally, Matplotlib is used to reproduce the output of the Brian example::

    from pylab import *
    subplot(3,1,1)
    plot(output_rate[0, :])
    subplot(3,1,2)
    plot(weights, '.')
    subplot(3,1,3)
    hist(weights, bins=20)
    show()