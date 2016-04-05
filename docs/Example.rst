************************************
Examples
************************************

This section provides a step-by-step description of some sample models provided in the ANNarchy package in the ``examples/`` directory.

* :doc:`example/NeuralField`:  a simple model using `neural field <http://www.scholarpedia.org/article/Neural_fields>`_ recurrent networks. This is a very simple rate-coded model without learning.

* :doc:`example/BarLearning`: an implementation of the bar learning problem, illustrating the synaptic plasticity in rate-coded networks. 

* :doc:`example/Izhikevich`: an implementation of the simple pulse-coupled network described in (Izhikevich, 2003). It shows how to build a simple spiking network without synaptic plasticity.

* :doc:`example/VogelsAbbott`: an implementation of the balanced network described in (Vogels and Abbott, 2005). It shows how to build a simple spiking network using integrate-and-fire neurons and sparse connectivity.

* :doc:`example/STDP`: an example using spike-timing dependent plasticity (STDP).
      
Other examples include: 

* ``examples/pyNN``: reproduction of several basic examples provided in the documentation of PyNN and Brian. They show mainly how to use the standard spiking neuron models.
* ``examples/refractoriness``: shows the effect of the refractory period on network behavior.
* ``examples/hybrid``: shows how to use hybrid populations (rate-coded and spiking).
* ``examples/image``: shows how to use the ``ImagePopulation`` and ``VideoPopulation`` classes of the ``image`` extension to clamp directly images and video streams into a rate-coded network.
* ``examples/hodgkin_huxley``: shows how to define a Hodgkin-Huxley neuron.

.. toctree::
    :maxdepth: 2

    example/NeuralField.rst
    example/BarLearning.rst
    example/Izhikevich.rst
    example/VogelsAbbott.rst
    example/STDP.rst
