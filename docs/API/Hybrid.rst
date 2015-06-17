****************************************
Hybrid networks
****************************************

Converting a rate-coded population to a spiking population requires connecting a ``PoissonPopulation`` (see :doc:`SpecificPopulation`) with the rate-coded one. 

Converting a spiking population with a rate-coded one requires the use of a ``DecodingProjection``, which can connected using any connector method available for ``Projection``.

Class DecodingProjection
===============================

.. autoclass:: ANNarchy.DecodingProjection
   :members:
