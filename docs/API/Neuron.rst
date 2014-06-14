**********************************
RateNeuron and SpikeNeuron
**********************************

Neurons are container objects for all information corresponding to a special neuron type. This encapsulation allows a higher readability of the code. Through derivation of ``ANNarchy.Neuron`` the user can define the neuron types he needs in his model. ANNarchy provides two specialised classes for the definition of neurons of different modeling types: rate-coded and spiking neurons.


* ``RateNeuron`` describes the behavior of a rate-coded neuron (see :doc:`../manual/RateNeuron`) 
* ``SpikeNeuron`` describes the behavior of a spiking neuron (see :doc:`../manual/SpikeNeuron`) 

Class RateNeuron
================

.. autoclass:: ANNarchy.RateNeuron
    :members:


Class SpikeNeuron
=================

.. autoclass:: ANNarchy.SpikeNeuron
    :members:
