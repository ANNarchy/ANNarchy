************************************
Examples
************************************

This section provides a step-by-step description of the sample models provided in the ANNarchy package in the ``examples/`` directory.

    * :doc:`example/NeuralField`: In ``examples/neural_field`` is implemented a simple model using `neural field <http://www.scholarpedia.org/article/Neural_fields>`_ recurrent networks. This is a very simple rate-coded model without learning.
    
    * :doc:`example/BarLearning`: In ``examples/bar_learning`` is an implementation of the bar learning problem. It illustrates the implementation of learning in rate-coded networks.

    * :doc:`example/Izhikevich`: In ``examples/izhikevich`` is an implementation of the simple pulse-coupled network described in (Izhikevich, 2003). It shows how to build a simple spiking network without synaptic plasticity.

.. toctree::
    :maxdepth: 2

    example/NeuralField.rst
    example/BarLearning.rst
    example/Izhikevich.rst
