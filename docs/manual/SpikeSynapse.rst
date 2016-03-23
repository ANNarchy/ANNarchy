***********************************
Spiking synapses
***********************************

Synapses in spiking networks differ from rate-coded synapses in that they are event-driven, i.e. the most important changes occur whenever a pre- or post-synaptic spike is emitted. For this reason, additional arguments have to be passed to the ``Synapse`` object.
   
Increase of conductance after a pre-synaptic spike
===================================================

In the simplest case, a pre-synaptic spike increases a ``target`` conductance value in the post-synaptic neuron. The rule defining how this conductance is modified has to be placed in the ``pre_spike`` argument of a ``Synapse`` object.

The default spiking synapse in ANNarchy is equivalent to:

.. code-block:: python

    DefaultSynapse = Synapse(
        parameters = "w=0.0",
        equations = "",
        pre_spike = """
            g_target += w
        """     
    ) 

The only thing it does is to increase the conductance ``g_target`` of the post-synaptic neuron (for example ``g_exc`` if the target is ``exc``) every time a pre-syanptic spike arrives at the synapse, proportionally to the synaptic efficiency ``w`` of the synapse. Note that ``w`` is implicitely defined in all synapses, you will never need to define it explicitely.

You can override this default behavior by providing a new ``Synapse`` object when building a ``Projection``. For example, you may want to implement a "fatigue" mechanism for the synapse, transciently reducing the synaptic efficiency when the pre-synaptic neuron fires too strongly. One solution would be to decrease a synaptic variable everytime a pre-synaptic spike  is received and increase the post-synaptic conductance proportionally to this value. When no spike is received, this ``trace`` variable should slowly return to its maximal value.

.. code-block:: python

    FatigueSynapse = Synapse(
        parameters = """
            tau = 1000 : post-synaptic # Time constant of the trace is 1 second
            dec = 0.05 : post-synaptic # Decrement of the trace
        """,
        equations = """
            tau * dtrace/dt + trace = 1.0 : min = 0.0
        """,
        pre_spike = """
            g_target += w * trace
            trace -= dec
        """     
    ) 
   
Each time a pre-synaptic spike occurs, the post-synaptic conductance is increased from ``w*trace``. As the baseline of ``trace`` is 1.0 (as defined in ``equations``), this means that a "fresh" synapse will use the full synaptic efficiency. However, after each pre-synaptic spike, trace is decreased from ``dec = 0.05``, meaning that the "real" synaptic efficiency can go down to 0.0 (the minimal value of trace) if the pre-synaptic neuron fires too often.

It is important here to restrict ``trace`` to positive values with the flags ``min=0.0``, as it could otherwise transform an excitatory synapse into an inhibitory one.

.. hint:: 

    It is obligatory to use the keyword ``g_target`` for the post-synaptic conductance. This value relates to the corresponding value in post-synaptic neuron: The ``target`` will be replaced with the projection's target (for example ``exc`` or ``inh``). So if you use this synapse in a projection with target = 'exc', the value of g_exc in post-synaptic neuron will be automatically replaced. 


Synaptic plasticity
==========================

In spiking networks, there are usually two methods to implement event-driven synaptic plasticity (see the entry on STDP at `Scholarpedia <http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity>`_):

* by using the difference in spike times between the pre- and post-synaptic neurons;
* by using online implementations.


Using spike-time differences
-----------------------------

A ``Synapse`` has access to two specific variables:

* ``t_pre`` corresponding to the time of the *last* pre-synaptic spike in milliseconds.

* ``t_post`` corresponding to the time of the *last* post-synaptic spike in milliseconds.
  
These times are relative to the creation of the network, so they only make sense when compared to each other or to ``t``.

Spike-timing dependent plasticity can for example be implemented the following way:

.. code-block:: python


    STDP = Synapse(
        parameters = """
            tau_pre = 10.0 : post-synaptic
            tau_post = 10.0 : post-synaptic
            cApre = 0.01 : post-synaptic
            cApost = 0.0105 : post-synaptic
            wmax = 0.01 : post-synaptic
        """,
        pre_spike = """
            g_target += w
            w = clip(w - cApost * exp((t_post - t)/tau_post) , 0.0 , wmax) 
        """,                  
        post_spike = """
            w = clip(w + cApre * exp((t_pre - t)/tau_pre) , 0.0 , wmax)
        """      
    ) 

* Every time a pre-synaptic spike arrives at the synapse (``pre_spike``), the post-synaptic conductance is increased from the current value of the synaptic efficiency. 

.. code-block:: python
    
    g_target += w

When a synapse object is defined, this behavior should be explicitely declared.

The value ``w`` is then decreased using a decreasing exponential function of the time elapsed since the last post-synaptic spike:

.. code-block:: python
    
    w = clip(w - cApost * exp((t_post - t)/tau_post) , 0.0 , wmax) 

The ``clip()`` global function is there to ensure that ``w`` is bounded between 0.0 and ``wmax``. As ``t >= t_post``, the exponential part is smaller than 1.0. The ``pre_spike`` argument therefore ensures that the synapse is depressed is a pre-synaptic spike occurs shortly after a post-synaptic one. "Shortly" is quantified by the time constant ``tau_post``, usually in the range of 10 ms.

* Every time a post-synaptic spike is emitted (``post_spike``), the value ``w`` is increased proportionally to the time elapsed since the last pre-synaptic spike:

.. code-block:: python
    
    w = clip(w + cApre * exp((t_pre - t)/tau_pre) , 0.0 , wmax)

This term defines the potentiation of a synapse when a pre-synaptic spike is followed immediately by a post-synaptic one: the inferred causality between the two events should be reinforced.

.. warning::

    Only the last pre- and post-synaptic spikes are accessible, not the whole history. Only **nearest-neighbor spike-interactions** are possible using ANNarchy, not temporal all-to-all interactions where the whole spike history is used for learning (see the entry on STDP at `Scholarpedia <http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity>`_).

    Some networks may not work properly when using this simulation mode. For example, whenever the pre-synaptic neurons fires twice in a very short interval and causes a post-synaptic spike, the corresponding weight should be reinforced twice. With the proposed STDP rule, it would be reinforced only once.

    It is therefore generally advised to use online versions of STDP.


Online version
---------------

The online version of STDP requires two synaptic traces, which are increased whenever a pre- resp. post-synaptic spike is perceived, and decay with their own dynamics in between. Using the same vocabulary as Brian, such an implementation would be:

.. code-block:: python

    STDP_online = Synapse(
        parameters = """
            tau_pre = 10.0 : post-synaptic
            tau_post = 10.0 : post-synaptic
            cApre = 0.01 : post-synaptic
            cApost = 0.0105 : post-synaptic
            wmax = 0.01 : post-synaptic
        """,
        equations = """
            tau_pre * dApre/dt = - Apre : event-driven
            tau_post * dApost/dt = - Apost : event-driven
        """,
        pre_spike = """
            g_target += w
            Apre += cApre 
            w = clip(w - Apost, 0.0 , wmax)
        """,                  
        post_spike = """
            Apost += cApost
            w = clip(w + Apre, 0.0 , wmax)
        """      
    ) 
    
The variables ``Apre`` and ``Apost`` are exponentially decreasing traces of pre- and post-synaptic spikes, as shown by the leaky integration in ``equations``. When a pre-synaptic spike is emitted, ``Apre`` is incremented, the conductance level of the post-synaptic neuron ``g_target`` too, and the synaptic efficiency is decreased proportionally to ``Apost`` (this means that if a post-synaptic spike was emitted shortly before, LTD will strongly be applied, while if it was longer ago, no major change will be observed). When a post-synaptic spike is observed, ``Apost`` increases and the synaptic efficiency is increased proportionally to ``Apre``. 

The effect of this online version is globally the same as the spike timing dependent version, except that the history of pre- and post-synaptic spikes is fully contained in the variables ``Apre`` and ``Apost``.

The ``event-driven`` keyword allows event-driven integration of the variables ``Apre`` and ``Apost``. This means the equations are not updated at each time step, but only when a pre- or post-synaptic spike occurs at the synapse. This is only possible because the two variables follow linear first-order ODEs. The event-driven integration method allows to spare a lot of computations if the number of spikes is not too high in the network.

Order of evaluation
--------------------

Three types of updates are potentially executed at every time step:

1. Pre-synaptic events, defined by ``pre_spike`` and triggered after each pre-synaptic spike, after a delay of at least ``dt``.
2. Synaptic variables defined by ``equations``.
3. Post-synaptic events, defined by ``post_spike`` and triggered after each post-synaptic spike, without delay.

These updates are conducted in that order at each time step. First, all spikes emitted in the previous step (or earlier if there are delays) are propagated to the corresponding synapses and influence variables there (especially conductance increases), then all synaptic variables are updated according to their ODE (after the neurons' equations are updated), then all neurons which have emitted a spike in the current step modify their synapses.

A potential problem arises when a pre-synaptic and a post-synaptic spike are emitted at the same time. STDP-like plasticity rules are usually not defined when the spike time difference is 0, as the two spikes can not be correlated in that case (the pre-spike can not possibly be the cause of the post-spike). 

By default, both event-driven updates (``pre_spike`` leading to LTD, ``post_spike`` leading to LTP) will be conducted when the spikes are emitted at the same time. This can be problematic for some plastic models, for example the ``simple_stdp`` example provided in the source code.

To avoid this problem, the flag ``unless_post`` can be specified in ``pre_spike`` to indicate that the corresponding variable should be updated after each pre-synaptic spike, **unless** the post-synaptic neuron also fired at the previous time step. Without even-driven integration, the online STDP learning rule would become:

.. code-block:: python

    STDP_online = Synapse(
        parameters = """
            tau_pre = 10.0 : post-synaptic
            tau_post = 10.0 : post-synaptic
            cApre = 0.01 : post-synaptic
            cApost = 0.0105 : post-synaptic
            wmax = 0.01 : post-synaptic
        """,
        equations = """
            tau_pre * dApre/dt = - Apre 
            tau_post * dApost/dt = - Apost 
        """,
        pre_spike = """
            g_target += w
            Apre += cApre : unless_post
            w = clip(w - Apost, 0.0 , wmax) : unless_post
        """,                  
        post_spike = """
            Apost += cApost
            w = clip(w + Apre, 0.0 , wmax)
        """      
    ) 


Continuous synaptic transmission
=================================

In some cases, synaptic transmission cannot be described in an event-driven framework. Synapses using the NMDA neurotransmitter are for example often modeled as non-linear synapses. Non-linear synapses can require the post-synaptic conductance to be a sum of synapse-specific variables, as for rate-coded neurons, and not simply incremented when a pre-synaptic spike occurs. NMDA synapses can be represented by two variables :math:`x(t)` and :math:`g(t)` following first-order ODEs:

.. math::
    
    \begin{aligned}
    \tau \cdot \frac{dx(t)}{dt} &= - x(t) \\
    \tau \cdot \frac{dg(t)}{dt} &= - g(t) +  x(t) \cdot (1 - g(t))
    \end{aligned}

When a pre-synaptic spike occurs, :math:`x(t)` is incremented by the weight :math:`w(t)`. However, it does not influence directly the post-synaptic neuron, as the output of a synapse is the signal :math:`g(t)`. The post-synaptic conductance is defined at each time :math:`t` as the sum over all synapses of the same type of their variable :math:`g(t)`:

.. math::

    g_\text{exc}(t) = \sum_{i=1}^{N_\text{exc}} g_i (t)


Such a synapse could be implemented the following way::

    NMDA = Synapse(
        parameters = """
        tau = 10.0 : postsynaptic
        """,
        equations = """
        tau * dx/dt = -x
        tau * dg/dt = -g +  x * (1 -g)
        """, 
        pre_spike = "x += w",
        psp = "g"
    )


The synapse defines a ``psp`` argument which means that the output of this synapse is non-linear and the post-synaptic conductance should be summed over this value (``g`` in this case). It is not possible to use the event-driven integration scheme for such non-linear synapses. 