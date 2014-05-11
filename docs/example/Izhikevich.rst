**************************
Simple pulse network
**************************

In ``examples/izhikevich`` is a simple model using a very simple spiking neuron. It consists of two 2D populations ``Excitatory`` and ``Inhibitory``, with excitory all-to-all connections between ``Excitatory`` and ``Inhibitory``, and inhibitory all-to-all connections within between ``Excitatory`` and ``Inhibitory``. Within both populations exists all-to-all connections excitatory in case of ``Excitatory`` respectively inhibitory within ``Inhibitory``.

You can simply try the network by typing::

    python Izhikevich.py
    
    
Description of the network
==========================

Defining the neuron type
========================