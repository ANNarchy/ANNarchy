************************************
Examples
************************************

This section provides a list of the sample models provided in the ``examples/`` directory of the source code.

Rate-coded networks
===================

* :doc:`example/NeuralField`:  a simple model using `neural field <http://www.scholarpedia.org/article/Neural_fields>`_ recurrent networks. This is a very simple rate-coded model without learning.

* :doc:`example/BarLearning`: an implementation of the bar learning problem, illustrating synaptic plasticity in rate-coded networks.

* :doc:`example/Image` and :doc:`example/Webcam`: shows how to use the ``ImagePopulation`` and ``VideoPopulation`` classes of the ``image`` extension to clamp directly images and video streams into a rate-coded network. Also demonstrate the ``convolution`` extension.

* :doc:`example/StructuralPlasticity`: a dummy example using structural plasticity.

* :doc:`example/MultipleNetworks`: shows how to use multiple networks and call ``parallel_run`` to run several networks in parallel.

Spiking networks
==================

**Simple networks**

* :doc:`example/Izhikevich`: an implementation of the simple pulse-coupled network described in (Izhikevich, 2003). It shows how to build a simple spiking network without synaptic plasticity.

* :doc:`example/GapJunctions`: an example using gap junctions.

* :doc:`example/HodgkinHuxley`: a single Hodgkin-Huxley neuron.

* A collection of Brain/PyNN/NEST model reproductions in the folder ``examples/pyNN``.

**Complex networks**

* :doc:`example/COBA`: an implementation of the balanced network described in (Vogels and Abbott, 2005). It shows how to build a simple spiking network using integrate-and-fire neurons and sparse connectivity.

* :doc:`example/STP`: an example of short-term plasticity based on the model of Tsodyks, Uziel and Markram (2000). *Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses*. The Journal of Neuroscience.

**With synaptic plasticity**

* :doc:`example/SimpleSTDP`: a simple example using spike-timing dependent plasticity (STDP).

* :doc:`example/Ramp`: an example of homeostatic STDP based on the model of Carlson, Richert, Dutt and Krichmar (2013). *Biologically plausible models of homeostasis and STDP: Stability and learning in spiking neural networks*. IJCNN.

Hybrid networks
==================

* :doc:`example/Hybrid`: a simple hybrid network with both rate-coded and spiking parts.


General
==================

* :doc:`example/BayesianOptimization`: a demo showing how to use ``hyperopt`` to search for hyperparameters of a model.

* :doc:`example/BasalGanglia`: a simple basal ganglia model to show how to use the ``tensorboard`` extension.

**List of notebooks:**

.. toctree::
    :maxdepth: 1

    example/NeuralField.ipynb
    example/BarLearning.ipynb
    example/Image.ipynb
    example/Webcam.ipynb
    example/MultipleNetworks.ipynb
    example/StructuralPlasticity.ipynb
    example/Izhikevich.ipynb
    example/GapJunctions.ipynb
    example/HodgkinHuxley.ipynb
    example/COBA.ipynb
    example/STP.ipynb
    example/SimpleSTDP.ipynb
    example/Ramp.ipynb
    example/Hybrid.ipynb
    example/BayesianOptimization.ipynb
    example/BasalGanglia.ipynb
