*******************************
Defining neurons
*******************************

A population consists of a certain number of neurons of identical type. Each neuron type has to be specified using the **RateNeuron** or **SpikeNeuron** classes depending on the desired framework. 

Rate coded neurons
===============================

Let's consider first a simple rate-coded neuron of the leaky-integrator type, which simply integrates the weighted sum of its excitatory inputs:

.. math::

    \tau \frac{d \text{mp}(t)}{dt} &= ( B - \text{mp}(t) ) + \sum_{i}^{\text{exc}} \text{rate}_{i} * w_{i} \\ 
           
    \text{rate}(t) & = ( \text{mp}(t) )^+
    
where :math:`mp(t)` represents the membrane potential of the neuron, :math:`\tau` the time constant of the neuron, :math:`B` its baseline firing rate, :math:`\text{rate}(t)` its instantaneous firing rate, :math:`i` an index over all excitatory synapses of this neuron, :math:`w_i` the efficiency of the synapse with the presynaptic neuron of firing rate :math:`\text{rate}_{i}`. 

It can be implemented in the ANNarchy framework with:

.. code-block:: python

    LeakyIntegratorNeuron = RateNeuron(
        parameters="""   
            tau = 10.0,
            baseline = -0.2,
        """
        equations = """
            tau * dmp / dt  = baseline - mp + sum(exc)
            rate = pos(mp)
        """
    )
    
The only required variable is ``rate``, which represents the instantaneous firing rate and will be used to propagate activity in the network. All other parameters and variables are freely decided by the user. More detailed examples can be found in the section `Examples <Examples.html>`_.

Custom functions can also be passed when creating the Neuron type:

.. code-block:: python

    LeakyIntegratorNeuron = RateNeuron(
        parameters="""   
            tau = 10.0,
            baseline = -0.2,
        """
        equations = """
            tau * dmp / dt  = baseline - mp + sum(exc)
            rate = sigmoid(mp)
        """,
        functions == """
            sigmoid(x) = 1.0 / (1.0 + exp(-x))
        """
    )

**Predefined attributes**

The ODE can depend on other parameters of the neuron (e.g. ``rate`` depends on ``mp``), but not on unknown names. ANNarchy already ensures the existence of the following variables and parameters for a neuron:

    * variable *rate*: instantaneous firing rate of the neuron. Initial value: 0.0.
    
    * variable *t*: current step of the simulation (incremented after each step of the simulation).
    
    * parameter *dt*: the discretization step, default is 1.0ms. 
    
**Weighted sum of inputs**

The ``sum()`` method of a neuron gives a direct access to the weighted sum of all inputs to the postsynaptic neuron separately by the connection type. These synapses are organized in a data structure called **Dendrite**. Some more information about the weighted sum can be found `here <Synapse.html#weighted-sum>`_.

.. warning:: 

    The connection type, e.g. *exc* or *inh*, need to match with the names used as a ``target`` parameter when creating a ``Projection``. If such a projection does not exist when the network is compiled, the weighted sum will be set to zero.

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
            dv/dt = 0.04 * v * v + 5*v + 140 -u + I : init = 0.0
            du/dt = a * (b*v - u) : init = -13.0
        """,
        spike = """
            v > T
        """,
        reset = """
            v = c
            u += d
        """
    )

**Spike condition**

The spike condition is a single constraint definition. You may use the different available comparison operators using all previously defined neuron variables.

The use of assignment statements or full ODEs will lead to an error. Furthermore the decision variable of the condition need to be placed on the **left** side.

**Reset**

Here you define the variables which should be set to certain values after a spike occured. Any assignment statements is allowed (``=``, ``+=``, etc), but the use of ODEs is not possible at this point, as the reset is performed only once at the end of the time step.

**Conductances**

Contrary to rate-coded neurons, spiking neurons use conductance variables to encode the received inputs, not weighted sums. In ANNarchy, the conductances are defined by ``g_`` followed by the target name. For example, if a population receives excitatory input (exc) from another one, you may access the conductance with:

.. code-block:: python

    dv/dt = v + g_exc

Exponentially decaying synapses (or any other function) must be specified after its usage in the membrane potential: 

.. code-block:: python

    Izhikevitch = SpikeNeuron(
    parameters="""
        a = 0.02
        b = 0.2
        c = -65.0
        d = 2.0
        T = 30.0
        tau = 3.0
    """,
    equations="""
        I = Normal(0.0,1.0)
        dv/dt = 0.04 * v * v + 5*v + 140 -u + I + g_exc: init = 0.0
        du/dt = a * (b*v - u) : init = -13.0
        tau * dg_exc/dt + g_exc = 0.0
    """,
    spike = """
        v > T
    """,
    reset = """
        v = c
        u += d
    """
