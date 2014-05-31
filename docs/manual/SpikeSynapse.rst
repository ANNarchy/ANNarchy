***********************************
Spike synapses
***********************************

Synapses in spiking networks differ from rate-coded synapses in that they are event-driven, i.e. the most important changes occur whenever a pre- or post-synaptic spike is emitted. For this reason, the interface of a ``SpikeSynapse`` slightly differs from a ``RateSynapse``.
   
Increase of conductance
=======================

In the simplest case, a presynaptic spike increases a ``target`` conductance value in the postsynaptic neuron. The rule defining how this conductance is modified can be placed in the pre-synaptic event section of a synapse:

.. code-block:: python

    pre_spike="""
        g_target += value
    """
    
Note that this is the default behaviour, only exceptions to this rule have to be implemented.

.. hint:: **current limitation**

    For the current implementation, it is obligatory to use the keyword ``g_target``. This value relates to the corresponding value in postsynaptic neuron: The ``target`` will be replaced with the projection's target (for example ``exc`` or ``inh``). So if you use this synapse in a projection with target = 'exc', the value of g_exc in postsynaptic neuron will be automatically replaced. In a further release it will be analogous to Brian.

Defining a learning rule
==========================

To define the learning rule you can describe the pre- and postsynaptic events separately in the synapse description (what happens when a pre- resp. post-synaptic spike is perceived at the corresponding synapse). The following example describes a basic implementation of STDP (Spike-Timing Dependent Plasticity), with the same formalism as in Brian:

.. code-block:: python

    SimpleLearn=SpikeSynapse(
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

Defining the refractory period
==============================

The refractory period is specified by the ''refractory'' parameter of ''SpikeNeuron''. As any other variable, it can be later modified for the whole population.

.. code-block :: python

    RefractoryNeuron = SpikeNeuron (
        parameters = """ ... """,
        equations = """ ... """,
        spike = """ ... """,
        reset = """ ... """,
        refractory = 5.0
    )