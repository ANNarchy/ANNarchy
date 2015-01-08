**********************************************
ANNarchy versions
**********************************************
     
Changelog
==========

**4.3.3:**

* Structural plasticity can be defined at the synaptic level.

**4.3.2:**

* Random distributions are available in synapses.
* Populations can be momentarily disabled/enabled.
* Recording can be done with a given period instead of every time step.
* Different modes for Spike2Rate conversion.
* Extension for shared projections (convolution/pooling) improved.

**4.3.1:** 

* CUDA implementation for rate-coded networks with small restrictions.

**4.3.1:** 

* Weight-sharing for convolutions and pooling.
* Possibility to create projections using a dense connection matrix or data saved in a file.
* A seed can be defined for the random number generators.

**4.3.0:** 

* Simplified internal structure for the generated code. 
* Major speed-up for compilation and execution times.
* Rate-coded projections can perform other functions than sum (min/max/mean) over the connected synapses.
  
**4.2.4:**

* Neuron and Synapse replace RateNeuron, SpikeNeuron, RateSynapse, SpikeSynapse in the main interface (kept for backward compatibility).
* Default PyNN neural models are now available (Izhikevich, IF_curr_exp, IF_cond_exp, IF_curr_alpha, IF_cond_alpha, HH_cond_exp, EIF_cond_alpha_isfa_ista, EIF_cond_exp_isfa_ista).
* Bug fixes for delays and spike propagation.
* Connectivity matrices (CSR) are more efficient.
* Several PyNN examples are added to the examples/ folder.

**4.2.3**

* Spike conditions can depend on several variables.
* The midpoint numerical method is added.
* Ability to choose globally the numerical method.

**4.2.2**

* The Euler implicit method is added.
* Systems of equations are now solved concurrentely.

**4.2.1:**

* Early-stopping of simulations (simulate_until).

**4.2.0:**

* Added hybrid populations (Spike2RatePopulaton, Rate2SpikePopulation).
* Individual save/load of populations and projections.
* PopulationViews can be used to build Projections.
* CSR objects are simplified.
* Default conductance behaviour (g_target = 0.0).
* Random distribution objects can take global parameters as arguments.
* min and max bounds can depend on other variables.
* Logical operators (and, or..) can be used in conditions.

**4.1.2:**

* Rate-coded neurons must output "r", not "rate". Synapses must use "w", not "value".
* Synapses can access pre-synaptic weighted sums (pre.sum(exc)).
* Created PoissonPopulation.

**4.1.1:**

* Connectivity matrices are now created in Cython.
* Recording methods have changed.
* Spiking neurons have a refractory period.
* Added clip() functions for emi-linear transfer functions.
* Added smoothed_firing_rate() method for visualizing spiking networks.
  
**4.1.0:**

* First stable release with both rate-coded and spiking networks.



Planned features
==================

* GPGPU implementation.



History
=========

A historical overview of the previous major versions:

* 1.0: Initial version, purely C++.
* 1.1: Management of exceptions.
* 1.3: Parallelization of the computation using openMP.
* 2.0: Optimized version with separated arrays for typed connections.
* 2.1: Parallelization using CUDA.
* 2.2: Optimized parallelization using openMP.
* 3.x: Python interface to the C++ core using Boost::Python.
* 4.x: Python-only version using Cython for the interface to the generated C++ code.  