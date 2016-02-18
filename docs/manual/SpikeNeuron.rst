===============================
Spiking neurons
===============================

Contrary to rate-coded neurons, the use of spiking neurons requires the additional definition of a spike condition (the criteria defining the emission of a spike, typically when the membrane potential exceeds a threshold) and reset equations, governing the evolution of all variables after a spike is emitted. 


Built-in neurons
================

ANNarchy provides standard spiking neuron models, similar to the ones defined in PyNN (`http://neuralensemble.org/docs/PyNN/reference/neuronmodels.html <http://neuralensemble.org/docs/PyNN/reference/neuronmodels.html>`_).

Their definition (parameters, equations) are described in :doc:`../API/SpecificNeuron`. The classes can be used directly when creating the populations (no need to instantiate them). Ex:

.. code-block:: python

    pop = Population(geometry = 1000, neuron = Izhikevich)


User-defined neurons
====================

Let's consider a simple leaky integrate-and-fire spiking neuron model (LIF) using a voltage-gated excitatory conductance:

.. math::

    \tau \cdot  \frac{ d v(t) }{ dt } = (E_r - v(t) ) + g_\text{exc}(t) \cdot (E_e -  v(t) )

where :math:`v(t)` is the membrane potential, :math:`\tau` is the membrane time constant (in milliseconds), :math:`E_r` the resting potential, :math:`E_e` the target potential for excitatory synapses and :math:`g_\text{exc}(t)` the total current induced by excitatory synapses.

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
        spike = "v > T",
        reset = "v = Er",
        refractory = 5.0
    )

As for rate-coded neurons, the parameters are defined in the ``parameters`` description, here globally for the population. ``equations`` contains the description of the ODE followed by the membrane potential. The additional information to provide is:

* ``spike`` : a boolean condition on a single variable (typically the membrane potential) deciding when a spike is emitted.
  
* ``reset`` : the modifications to the neuron's variables after a spike is emitted (typically, clamping the membrane potential to its reset potential).

* ``refractory``: optionally a refractory period in ms.

Spike condition
----------------

The spike condition is a single constraint definition. You may use the different available comparison operators (>, <,  ==, etc) on a **single** neuron variable, using as many parameters as you want.

The use of assignment statements or ODEs will lead to an error. Conditional statements can be used. Example: 

.. code-block:: python

    parameters="""
        ...
        T = -45.0 
    """,
    equations="""
        prev_v = v
        noise = Uniform (-5.0, 5.0)
        tau*dv/dt = E - v + g_exc
    """,
    spike = """
        (v > T + noise) and (prev_v < T + noise)
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
        spike = " ... ",
        reset = " ... "
    )

Incoming spikes increase ``g_exc`` and can provoke a post-synaptic spike at the next step, but leave no trace beyond that point.

* Most models however use **exponentially decaying synapses**, where the conductance decays with a short time constant after a spike is received. This behavior should be explicitely specified in the neuron's equations: 

.. code-block:: python

    LIF = Neuron(
        parameters=""" ... """,
        equations="""
            tau * dv/dt = (Er - v) + g_exc *(Ee- v) : init = 0.0
            tau_exc * dg_exc/dt = - g_exc
        """,
        spike = " ... ",
        reset = " ... "
    )

``g_exc`` is increased by incoming spikes, and slowly decays back to 0.0 until the next spikes arrive.

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

The ``refractory`` argument can be a floating value or the name of a parameter/variable (string).

If ``dt = 0.1``, this means that the ``equations`` will not be evaluated for 50 consecutive steps after a spike is emitted, except for the conductances (starting with ``g_``) which are evaluated normally during the refractory period (the neuron is not "deaf", it only is frozen in a refractory state). 

``refractory`` becomes an attribute of a spiking ``Population`` object, so it can be set specifically for a population even when omitted in the neuron definition:

.. code-block:: python

    LIF = Neuron (
        parameters = " ... ",
        equations = " ... ",
        spike = " ... ",
        reset = """ 
            v = c
            u += d
        """
    )

    pop = Population(geometry = 1000, neuron = LIF)
    pop.refractory = Uniform(1.0, 10.0)

It can be either a single value, a ``RandomDistribution`` object or a Numpy array of the same size/geometry as the population.

Instantaneous firing rate
---------------------------

**Method 1: ISI**

Spiking neurons define an additional variable ``t_last`` which represents the timestamp (in ms) of the last emitted spike (updated at the end of the ``reset`` statement). The time elapsed since the last spike is then ``t - t_last``.

This can be used to update the instantaneous firing rate of a neuron, by inverting the inter-spike interval (ISI) during the ``reset`` statement following the emission of a spike::

    neuron = Neuron(
        parameters = "tau = 20.0; tauf = 1000.",
        equations = """
            tau * dv/dt + v = ...
            tauf * df/dt = -f
        """,
        spike = "v > 1.0",
        reset = """
            v = 0.0
            f = 1000./(t - t_last)
        """
    )

Here, a leaky integrator on f is needed to 1) smooth the firing rate and 2) slowly decay to 0 when the neuron stops firing. This method reflects very fast changes in the firing rate, but is also very sensible to noise.


**Method 2: Window**

A more stable way to compute the firing rate of a neuron is to count at each time step the number of spikes emitted during a sliding temporal window (of 100 ms or 1s for example). By default, spiking neurons only record the time of the last spike they emitted (``t_last``), so this mechanism has to be explicitely enabled before compilation by calling the ``compute_firing_rate()`` method of the desired population::

    pop = Population(100, Izhikevich)
    pop.compute_firing_rate(window=1000.0)
    compile()

The ``window`` argument represents the period in milliseconds over which the spikes will be counted. The resulting firing rate (in Hz) will be stored in the local variable ``r`` (as for rate-coded neurons), which can be accessed by the neuron itself or by incoming and outgoing synapse (``pre.r`` and ``post.r``). 

If the method has not been called, the variable ``r`` of a spiking neuron will be constantly 0.0.
