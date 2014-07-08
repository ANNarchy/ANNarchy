===============================
Spiking neurons
===============================

Contrary to rate-coded neurons, the use of spiking neurons requires the additional definition of a spike condition (the criteria defining the emission of a spike, typically when the membrane potential exceeds a threshold) and reset equations, governing the evolution of all variables after a spike is emitted. 

Let's consider a simple leaky integrate-and-fire spiking neuron model (LIF):

.. math::

    \tau \cdot  \frac{ d v(t) }{ dt } = (E_r - v(t) ) + g_\text{exc}(t) \cdot (E_e -  v(t) )

where :math:`v(t)` is the membrane potential, :math:`\tau` is the membrane time constant (in milliseconds), :math:`E_r` the resting potential, :math:`E_e` the target potential for excitatory synapses and :math:`g_\text{exc}(t)` the total condutance of excitatory synapses.

This neural model can be defined in ANNarchy by:

.. code-block:: python

    LIF = Neuron(
        parameters="""
            tau = 10.0  : population
            Er = -60.0  : population
            Ee = 0.0    : population
            T = -45.0   : population
        """,
        equations="""
            tau * dv/dt = (Er - v) + g_exc *(Ee- v) : init = 0.0
        """,
        spike = """
            v > T
        """,
        reset = """
            v = Er
        """,
        refractory = 1.0
    )

As for rate-coded neurons, the parameters are defined in the ``parameters`` description, here globally for the population. ``equations`` contains the description of the ODE followed by the membrane potential. The additional information to provide is:

* ``spike`` : a boolean condition on a single variable (typically the membrane potential) deciding when a spike is emitted.
  
* ``reset`` : the modifications to the neuron's variables after a spike is emitted (typically, clamping the membrane potential to its reset potential).

Spike condition
----------------

The spike condition is a single constraint definition. You may use the different available comparison operators (>, <,  ==, etc) on a **single** neuron variable, using as many parameters as you want.

The use of assignment statements or full ODEs will lead to an error. Furthermore the decision variable of the condition needs to be placed alone on the **left** side.

Example: 

.. code-block:: python

    parameters="""
        ...
        T = -45.0 
    """,
    equations="""
        noise = Uniform (-5.0, 5.0)
        ...
    """,
    spike = """
        v > T + noise
    """

Reset
------

Here you define the variables which should be set to certain values after a spike occured. Any assignment statements is allowed (``=``, ``+=``, etc), but the use of ODEs is not possible, as the reset is performed only once at the end of the time step.

Example: 

.. code-block:: python

    reset = """
        v = Er 
        u += 0.1 
    """
  

Conductances
------------

Contrary to rate-coded neurons, spiking neurons use conductance variables to encode the received inputs, not weighted sums. In ANNarchy, the conductances are defined by ``g_`` followed by the target name. For example, if a population receives excitatory input (target ``exc``) from another one, you can access the total conductance provoked by ``exc`` spikes with:

.. code-block:: python

    tau * dv/dt + v = g_exc

The dynamics of the conductance can be specified after its usage in the membrane potential equation.

* The default behaviour for conductances is an **instantaneous reset** (or infinitely fast exponential decay). In practice, this means that all incoming spikes are summed up (weighted by the synaptic efficiency) at the beginning of a simulation step, and the resulting conductance is reset to 0.0 at the end of the step. This default behaviour is equivalent to :
  

.. code-block:: python

    LIF = Neuron(
        parameters=""" ... """,
        equations="""
            tau * dv/dt = (Er - v) + g_exc *(Ee- v) : init = 0.0
            g_exc = 0.0
        """,
        spike = """ ... """,
        reset = """ ... """
    )

Incoming spikes increase ``g_exc`` and can provoke a postsynaptic spike at the next step, but leave no trace beyond.

* Most models however use **exponentially decaying synapses**, where the conductance decays with a short time constant after a spike is received. This behavior should be explicitely specified in the neuron's equations: 

.. code-block:: python

    LIF = Neuron(
        parameters=""" ... """,
        equations="""
            tau * dv/dt = (Er - v) + g_exc *(Ee- v) : init = 0.0
            tau_exc * dg_exc/dt = - g_exc
        """,
        spike = """ ... """,
        reset = """ ... """
    )

``g_exc`` is increased by incoming spikes, and slowly decays back to 0.0 until the next spikes arrive.

.. note::

    The conductance's dynamics should be placed **after** updating the membrane potential, otherwise the value of ``g_exc`` would be already scaled down. 

Refractory period
-----------------

The refractory period in milliseconds is specified by the ``refractory`` parameter of ``Neuron``. 

.. code-block:: python

    LIF = Neuron (
        parameters = """ ... """,
        equations = """ ... """,
        spike = """ ... """,
        reset = """ 
            v = c
            u += d
        """,
        refractory = 5.0
    )

If ``dt = 1.0``, this means that the ``reset`` function will be called for 5 consecutive steps after a spike is emitted, in addition to the step where the spike was emitted. The equations will be evaluated normally, so ``g_exc`` will not "miss" incoming spikes during this period, only ``v`` will be stuck to ``c`` and ``u`` incremented 6 times altogether. 

``refractory`` becomes an attribute of a spiking ``Population`` object, so it can be set specifically for a population even when omitted in the neuron definition:

.. code-block:: python

    LIF = Neuron (
        parameters = """ ... """,
        equations = """ ... """,
        spike = """ ... """,
        reset = """ 
            v = c
            u += d
        """
    )

    pop = Population(geometry = 1000, neuron = LIF)
    pop.refractory = Uniform(1.0, 10.0)

It can be either a single value, a ``RandomDistribution`` object or a Numpy array of the same size/geometry as the population.

If you want only a subpart of the ``reset`` statements to be executed during the refractory period, you can use the ``unless_refractory`` flag. Statements flagged with ``unless_refractory`` will only be executed once just after a spike is emitted, but not during the refractory period. In the example above, it would indeed make more sense to define ``u`` as non-refractory, as the increment should be executed only once:

.. code-block:: python

    LIF = Neuron (
        parameters = """ ... """,
        equations = """ ... """,
        spike = """ ... """,
        reset = """ 
            v = c
            u += d : unless_refractory
        """,
        refractory = 5.0
    )



