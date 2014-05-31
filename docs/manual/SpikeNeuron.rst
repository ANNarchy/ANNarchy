===============================
Spiking neurons
===============================

Contrary to rate-coded neurons, the use of spiking neurons requires the aditional definition of a spike condition (the criteria defining the emission of a spike, typically when the membrane potential exceeds a threshold) and reset equations, governing the evolution of all variables after a spike is emitted. Let's consider a simple spiking neuron model as proposed by Izhikevich:

.. math::

    \frac{ d \text{u}_i(t) }{ dt } = a * ( \text{b} * \text{v} - \text{u}_i(t) )

    \frac{ d \text{v}_i(t) }{ dt } = 0.04 * \text{v}_i(t)^2 + 5 * \text{v}_i(t) + 140 - \text{u}_i(t) + \text{I}_i(t)

This neural model can be defined in ANNarchy by:

.. code-block:: python

    Izhikevitch = SpikeNeuron(
        parameters="""
            a = 0.02
            b = 0.2
            c = -65.0
            d = 2.0
            T = 30.0
        """,
        equations="""
            I = Normal(0.0,1.0) 
            dv/dt = 0.04 * v**2 + 5.*v + 140.0 -u + I : init = 0.0
            du/dt = a * (b*v - u) : init = -13.0
        """,
        spike = """
            v > T
        """,
        reset = """
            v = c
            u += d
        """,
        refractory = 1.0
    )

**Spike condition**

The spike condition is a single constraint definition. You may use the different available comparison operators using the previously defined neuron variables.

The use of assignment statements or full ODEs will lead to an error. Furthermore the decision variable of the condition needs to be placed on the **left** side.

**Reset**

Here you define the variables which should be set to certain values after a spike occured. Any assignment statements is allowed (``=``, ``+=``, etc), but the use of ODEs is not possible at this point, as the reset is performed only once at the end of the time step.

**Conductances**

Contrary to rate-coded neurons, spiking neurons use conductance variables to encode the received inputs, not weighted sums. In ANNarchy, the conductances are defined by ``g_`` followed by the target name. For example, if a population receives excitatory input (exc) from another one, you may access the conductance with:

.. code-block:: python

    dv/dt = v + g_exc

The dynamics of the conductance must be specified after its usage in the membrane potential equation.

* An instantaneous synaptic conductance  should be set to 0.0 at the end of ``equations``:

.. code-block:: python

    Izhikevitch = SpikeNeuron(
        parameters=""" ... """,
        equations="""
            I = Normal(0.0,1.0)
            dv/dt = 0.04 * v * v + 5*v + 140 -u + I + g_exc: init = 0.0
            du/dt = a * (b*v - u) : init = -13.0
            g_exc = 0.0
        """,
        spike = """ ... """,
        reset = """ ... """
    )

Incoming spikes increase ``g_exc`` and may provoke a postsynaptic spike at the next step, but leave no trace beyond.

* Exponentially decaying synapses should be also specified: 

.. code-block:: python

    Izhikevitch = SpikeNeuron(
        parameters=""" ... tau = 5.0 """,
        equations="""
            I = Normal(0.0,1.0)
            dv/dt = 0.04 * v * v + 5*v + 140 -u + I + g_exc: init = 0.0
            du/dt = a * (b*v - u) : init = -13.0
            tau * dg_exc/dt = - g_exc 
        """,
        spike = """ ... """,
        reset = """ ... """
    )

``g_exc`` is increased by incoming spikes, and slowly decays back to 0.0 until the next spikes arrive.

.. warning::

    If you forget to update the conductances after the equations, they may increase indefinitely!

**Refractory period**

The refractory period is specified by the ``refractory`` parameter of ``SpikeNeuron``. As any other variable, it can be later modified for the whole population.

.. code-block :: python

    RefractoryNeuron = SpikeNeuron (
        parameters = """ ... """,
        equations = """
            I = Normal(0.0,1.0)
            dv/dt = 0.04 * v * v + 5*v + 140 -u + I + g_exc: init = 0.0
            du/dt = a * (b*v - u) : init = -13.0
            tau * dg_exc/dt = - g_exc 
        """,
        spike = """
        v > T
        """,
        reset = """ 
            v = c
            u += d
        """,
        refractory = 5.0
    )

If ``dt = 1.0``, this means that the ``reset`` function will be called for 5 consecutive steps after a spike is emitted, in addition to the step where the spike was emitted. The equations will be evaluated normally, so ``g_exc`` will not "miss" incoming spikes during this period, only ``v`` will be stuck to ``c`` and ``u`` incremented 6 times altogether. 

