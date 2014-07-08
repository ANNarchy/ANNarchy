**********************************
Synapse
**********************************

The class ``Synapse`` is used to describe the behavior of a synapse (parameters, equations...).

It guesses automatically its type (rate-coded or spiking) based on the pre-synaptic population used in the Projection. For backward compatibility, you may also use: 

* ``RateSynapse`` describes the behavior of a rate-coded synapse (see :doc:`../manual/RateSynapse`) 
* ``SpikeSynapse`` describes the behavior of a spiking synapse (see :doc:`../manual/SpikeSynapse`) 


Class Synapse
=================

.. autoclass:: ANNarchy.Synapse
    :members:

Class RateSynapse
=================

.. autoclass:: ANNarchy.RateSynapse
    :members:


Class SpikeSynapse
==================

.. autoclass:: ANNarchy.SpikeSynapse
    :members:
