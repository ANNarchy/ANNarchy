***********************************
Spike synapses
***********************************

Synapses in spiking networks differ from rate-coded synapses in that they are event-driven, i.e. the most important changes occur whenever a pre- or post-synaptic spike is emitted. For this reason, the interface of a ``SpikeSynapse`` slightly differs from a ``RateSynapse``.
   
Increase of conductance after a presynaptic spike
==================================================

In the simplest case, a presynaptic spike increases a ``target`` conductance value in the postsynaptic neuron. The rule defining how this conductance is modified can be placed in the ``pre_spike`` argument of a ``SpikeSynapse`` object.

The default spiking synapse in ANNarchy is equivalent to:

.. code-block:: python

    DefaultSynapse = SpikeSynapse(
        parameters = "",
        equations = "",
        pre_spike = """
            g_target += w
        """     
    ) 

The only thing it does is to increase the conductance ``g_target`` of the postsynaptic neuron (for example ``g_exc`` if the target is ``exc``) every time a pre-syanptic spike arrives at the synapse, proportionally to the synaptic efficiency ``w`` of the synapse. 

You can override this default behavior by providing a new ``SpikeSynapse`` object when building a ``Projection``. For example, you may want to implement a "fatigue" mechanism for the synapse, transciently reducing the synaptic efficiency when the pre-synaptic neuron fires too strongly. One solution would be to decrease a synaptic variable everytime a pre-synaptic spike  is received and increase the post-synaptic conductance proportionally to this value. When no spike is received, this ``trace`` variable should slowly return to its maximal value.

.. code-block:: python

    FatigueSynapse = SpikeSynapse(
        parameters = """
            tau = 1000 : postsynaptic # Time constant of the trace is 1 second
            dec = 0.05 : postsynaptic # Decrement of the trace
        """,
        equations = """
            tau * dtrace/dt + trace = 1.0 : min = 0.0
        """,
        pre_spike = """
            g_target += w * trace
            trace -= dec
        """     
    ) 
   
Each time a pre-synaptic spike occurs, the postsynaptic conductance is increased from ``w*trace``. As the baseline of ``trace`` is 1.0 (as defined in ``equations``), this means that a "fresh" synapse will use the full synaptic efficiency. However, after each pre-synaptic spike, trace is decreased from ``dec = 0.05``, meaning that the "real" synaptic efficiency can go down to 0.0 (the minimal value of trace) if the pre-synaptic neuron fires too much.

It is important here to restrict ``trace`` to positive values with the flags ``min=0.0``, as it could otherwise transform an excitatory synapse into an inhibitory one...

.. hint:: 

    It is obligatory to use the keyword ``g_target`` for the post-synaptic conductance. This value relates to the corresponding value in postsynaptic neuron: The ``target`` will be replaced with the projection's target (for example ``exc`` or ``inh``). So if you use this synapse in a projection with target = 'exc', the value of g_exc in postsynaptic neuron will be automatically replaced. 

.. note::

    The ``psp`` argument of a ``RateSynapse`` is not valid anymore for a ``SpikeSynapse``.

Synaptic plasticity
==========================

In spiking networks, there are usually two ways to implement synaptic plasticity (see the entry on STDP on `Scholarpedia <http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity>`_):

* by using the difference in spike times between the pre- and post-synaptic neurons;
* by using online implementations.
  
.. Although the second approach should be preferred as it fits better the ANNarchy underlying structure and allows for efficient parallelization, both approaches are possible.

Using spike-time differences
-----------------------------

A ``SpikeSynapse`` has access to two specific variables:

* ``t_pre`` corresponding to the time of the *last* pre-synaptic spike in milliseconds.

* ``t_post`` corresponding to the time of the *last* post-synaptic spike in milliseconds.

.. code-block:: python

    STDP = SpikeSynapse(
        parameters = """
            tau_pre = 10.0 : postsynaptic
            tau_post = 10.0 : postsynaptic
            Apre = 1.0 : postsynaptic
            Apost = 1.0 : postsynaptic
        """,
        equations = "",
        pre_spike = """
            g_target += w
            w -= Apre * exp((t_post - t)/tau_pre)
        """,                  
        post_spike = """
            w += Apost * exp((t_pre - t)/tau_post)
        """      
    ) 



Online versions
---------------

To define a learning rule, you have to describe the pre- and postsynaptic events separately in the synapse description (what happens when a pre- or. post-synaptic spike is perceived at the corresponding synapse). 



The following example describes a basic implementation of STDP (Spike-Timing Dependent Plasticity), with the same formalism as in Brian:

.. code-block:: python

    STDP = SpikeSynapse(
        parameters = """
            tau_pre = 5 : postsynaptic
            tau_post = 5 : postsynaptic
            cApre = 1 : postsynaptic
            cApost = -1 : postsynaptic
        """,
        equations = """
            tau_pre * dApre/dt = -Apre
            tau_post * dApost/dt = -Apost
        """,
        pre_spike = """
            Apre += cApre
            g_target += value
            value += Apost
        """,                  
        post_spike = """
            Apost += cApost
            value += Apre
        """      
    ) 
    
The parameters are declared postsynaptic because they are the same for all synapses in the projection. The variables ``Apre`` and ``Apost`` are exponentially decreasing traces of pre- and post-synaptic spikes, as shown by the leaky integration in ``equations``. When a presynaptic spike is emitted, ``Apre`` is incremented, the conductance level of the postsynaptic neuron ``g_target`` too, and the synaptic efficiency is decreased proportionally to ``Apost`` (this means that if a post-synaptic spike was emitted shortly before, LTD will strongly be apllied, while if it was longer ago, no major change will be observed). When a post-synaptic spike is observed, ``Apost`` increases and the synaptic efficiency is increased proportionally to ``Apre``. 

