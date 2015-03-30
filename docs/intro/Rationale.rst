**********************************
Why another neural simulator?
**********************************

Existing simulators
============================

There is already a huge variety of neural simulators available, among which:

* `NEURON <http://www.neuron.yale.edu/neuron>`_, a simulation environment for modeling individual neurons and networks of neurons developed by Ted Carnevale and Michael Hines.
* `Brian <http://briansimulator.org>`_, a pure Python simulator written by Romain Brette and Dan Goodman.
* `NEST <http://www.nest-initiative.org>`_ (Neural Simulation Technology), a simulation system for large networks of biologically realistic spiking point-neurons, written in C++ with a Python interface by Marc-Oliver Gewaltig and Markus Diesmann.
* `PCSIM <http://www.lsm.tugraz.at/pcsim>`_ (Parallel neural Circuit SIMulator), a tool for simulating heterogeneous networks composed of different model neurons and synapses, written in C++ with a Python interface by Dejan Pecevski, Thomas Natschläger and Klaus Schuch.      
   
Although these simulators have different structures, scopes, interfaces, there exists some initiatives such as NeuroML or PyNN which provide a common interface allowing to simulate the same description of a model on different simulators. A lot of efforts have also been made recently on the parallel simulation of neural networks on multicore CPUs using MPI or openMP, or even GPGPUs (general-purpose graphics processing units) using CUDA or OpenCL. 

The most recent of these neural simulators focus mainly on a specific type of neural model, the spiking point-neuron, which present the interesting property of describing rather accurately the dynamics of biological neurons, while reducing the communication between neurons to the emission of a spike, a boolean value that happens rather rarely (neurons in the cerebral cortex typically fire at 3-10 Hz, while the dynamical equations underlying their dynamics in a neural model are usually simulated with a time step of 1 ms). This property is fully exploited for the parallel simulation of spiking neural networks, where communication between units is a very expensive operation. 

However, there is another type of artificial neurons which is widely used in computational neuroscience, especially when dealing with higher-order cognitive structures: the mean-firing rate neuron. This type of neuron only consider the instantaneous firing rate of a neuron instead of its precise spiking activity. Although some information is lost due to this abstraction (synchrony, spikes arrival order, etc), this type of neuron is very useful for cognitive modeling thanks to the powerful learning rules available in this framework. Nevertheless, recent neural simulators tend to neglect this important formalism.

Spiking neural networks
================================

Spiking neurons model the dynamics of biological neurons by neglecting the spatial structure of the cell (point-neuron) and modeling the evolution of their membrane potential with respect to their inputs (injected currents, synapses). The simplest type of spiking neuron is the *leaky integrate-and-fire* (LIF) neuron:

.. math::

     \tau \frac{d V(t)}{dt} &= (V_m - V(t))  + R * I(t) \\
         
     \text{if} &\quad V(t) > V_{th} : \qquad V(t) \gets V_{\text{reset}} \quad \text{and emission of a spike.}
     
where :math:`V(t)` denotes the membrane potential of the neuron, :math:`V_m` its equilibrium potential (the potential the cell would have if no current is injected), :math:`I(t)` the total current injected through the membrane, :math:`\tau` the time constant of the membrane, :math:`R` the resistance of the membrane, :math:`V_{th}` the threshold potential for the emission of a spike and :math:`V_{\text{reset}}` the reset potential after the emission.

The ordinary differential equation (ODE) which governs the evolution of the membrane potential is usually discretized using the Euler or Runge-Kutta methods, with a time step of 1 ms or less. When the membrane potential exceeds the threshold, a spike is emitted and sent to all neurons forming synapses with it, thereby increasing the input current :math:`I(t)` for a brief period of time depending on the synapse type. The input current due to synaptic activation is usually a weighted sum of post-synaptic currents (modelled as a Dirac, exponentially decaying or alpha function of the time elapsed since the pre-synaptic neuron has spiked):

.. math::

    I(t) = \sum_{i=1}^{N_{\text{synapses}}} w_i (t) \cdot f(t - t^{\text{spike}}_i )

Mean-firing rate neural networks
=============================================

Mean-firing rate neurons do not simulate the emission of single spikes, but rather compute directly their instantaneous firing rate (number of spikes per second, also expressed in Hz). 

The instantaneous firing rate of a neuron is updated through:

.. math::

    \tau \frac{dv(t)}{dt} &= ( B - v(t)) + I(t) \\ 
           
    r(t) & = f( v(t) )
    
where :math:`\tau` is the time constant of the neuron, :math:`v(t)` its membrane potential, :math:`B` its baseline, :math:`r(t)` its instantaneous firing rate, :math:`f(\cdot)` a transfer function and :math:`I(t)` the weighted sum of its inputs:

.. math::

    I(t) = \sum_{i=1}^{N_{\text{synapses}}} w_i (t) \cdot r_i (t - d)


Mean-firing rate networks require much more intercommunication between neurons than spiking neurons, because the weighted sum has to be computed at every time step for all incoming synapses. In spiking networks, the weighted sum can be only updated when some pre-synaptic emits a spike, which is a relatively rare event. 

Goal of the ANNarchy simulator
=======================================

The goal of ANNarchy is to provide a simulator that is equally optimized for both types of networks and allows for mixtures of the two frameworks (hybrid networks). Computations are for the moment distributed using either OpenMP on shared memory CPU-based systems or CUDA on GPU-based systems. 

It is particularly intended for cognitive modeling, where emphasis is put on the global function performed by a network of heterogeneous populations rather than local interactions within a population. It is designed to integrate easily external libraries (either in Python or C++) linking the network to real-time applications such as webcams, virtual reality environments or robots, as simulations are not expected to have a fixed duration, contrary to other simulators.







