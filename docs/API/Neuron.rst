**********************************
Neuron
**********************************

Neurons are container objects for all information corresponding to a special neuron type. This encapsulation allows a higher readability of the code. Through derivation of ``ANNarchy.Neuron``, the user can define the neuron types he needs in his model. 

The type of the neuron (rate-coded or spiking) depends on the presence of the ``'spike'`` argument. For backward compatibility, you may also use the following classes:

* ``RateNeuron`` describes the behavior of a rate-coded neuron (see :doc:`../manual/RateNeuron`) 
* ``SpikeNeuron`` describes the behavior of a spiking neuron (see :doc:`../manual/SpikeNeuron`) 

Class Neuron
================

.. autoclass:: ANNarchy.Neuron
    :members:

Class RateNeuron
================

.. autoclass:: ANNarchy.RateNeuron
    :members:


Class SpikeNeuron
=================

.. autoclass:: ANNarchy.SpikeNeuron
    :members:
