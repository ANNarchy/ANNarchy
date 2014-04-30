**************************
Simple pulse network
**************************

In ``examples/izhikevich`` is a simple model using a very simple spike model. It consists of two 2D populations ``Excitatory`` and ``Inhibitory``, with excitory all-to-all connections between ``Excitory`` and ``Inhibitory``, and inhibitory all-to-all connections within between ``Excitory`` and ``Inhibitory``. Within both populations exists all-to-all connections excitatory in case of ``Excitatory`` respectively inhibitory within ``Inhibitory``.

You can simply try the network by typing::

    python Izhikevich.py
    
    
.. image:: ../_static/izhikevich.png
    :align: center
    :width: 80%
    
Model overview
--------------------
    
The first population ``Excitatory`` consists of 800 neurons. The ``Inhibitory`` population has 200 neurons. Both populations uses the same equation set:

.. math::

    \frac{ d \text{u}_i(t) }{ dt } = a * ( \text{b} * \text{v} - \text{u}_i(t) )

    \frac{ d \text{v}_i(t) }{ dt } = 0.04 * \text{v}_i(t)^2 + 5 * \text{v}_i(t) + 140 - \text{u}_i(t) + \text{I}_i(t)

wherease the injection current \text{I} is computed as the following:

.. math::

    \text{I}_i(t) = \text{sum}_\text{exc} + \text{sum}_\text{inh} + \text{noise}

where :math:`\text{u}_i(t)` is the neuron's spiking threshold, :math:`\text{v}_i(t)` its membrane potential, :math:`\text{I}_i(t)` its injection current.

Defining the neurons
--------------------------

There are two different ODEs for the neurons, so we need to define two **Neuron** objects: ``InputNeuron`` and ``NonLinearNeuron`` (for example). To understand how to do this, please read carefully the section `Defining a Neuron <ImplementingNeuron.html>`_.

**Izhikevich**

``Izhikevich`` could be defined as the following:

.. code-block:: python

    Izhikevitch = Neuron(
        I_in = Variable(init=0.0),
        noise_scale = 5.0,
        I = Variable(init=0.0, eq = "I = sum(exc) + sum(inh) + noise*noise_scale"),
        noise = Variable(eq=Normal(0.0,1.0)),
        a = Variable(init=0.02),
        b = Variable(init=0.2),
        c = Variable(init=-65.0),
        d = Variable(init=2.0),
        u = Variable(init=-65.*0.2, eq="du/dt = a * (b*v - u)"),
        v = SpikeVariable(eq="dv/dt = 0.04 * v * v + 5*v + 140 -u + I", threshold= 30.0, init=-65.0, reset=['v = c', 'u = u+d']),
        order = ['I', 'v','u']
    )

**Inhibitory and Excitatory**

These two neuron types are different through their parameterization. One could define the following set (adapted from Izhikevich, 2003)

.. code-block:: python

    Excitatory = Population(name='Excitory', geometry=(nb_exc_neurons), neuron=Izhikevitch)
    re = np.random.random(nb_exc_neurons)
    Excitatory.c = -65.0 + 15.0*re**2
    Excitatory.d = 8.0 - 6.0*re**2
    
    Inhibitory = Population(name='Inhibitory', geometry=(nb_inh_neurons), neuron=Izhikevitch)
    ri = np.random.random(nb_inh_neurons)
    Inhibitory.noise_scale=2.0
    Inhibitory.b = 0.25 - 0.05*ri
    Inhibitory.a = 0.02 + 0.08*ri
    Inhibitory.u = (0.25 - 0.05*ri) * (-65.0) # b * -65

Defining the synapse
--------------------------

As synapse we use a straight-forward approach: the postsynaptic potential is equal to the amount of presynaptic spikes. One may implement it like the following:

.. code-block:: python

    Simple = Synapse(
        psp = Variable(init=0, eq = "psp = if t is (t_spike+1) then value else 0.0")
    )