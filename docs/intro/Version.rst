**********************************************
ANNarchy versions
**********************************************

Major versions
==============================================

A historical overview:

    * 1.0: Initial version, purely C++.
    * 1.1: Management of exceptions.
    * 1.3: Parallelization of the computation using openMP.
    * 2.0: Optimized version with separated arrays for typed connections.
    * 2.1: Parallelization using CUDA.
    * 2.2: Optimized parallelization using openMP.
    * 3.x: Python interface to the C++ core using Boost::Python.
    * 4.x: Python-only version using Cython for the interface to the generated C++ code.

Minor releases in the current major
==============================================

The current major release 4.x subdivides in the current versions.

    * 4.0.alpha   Python-only version using Cython for the interface to the generated C++ code.
    * 4.0.beta    Improved connection pattern handling and interface changes.
    * 4.0.gamma   Introduction of spike coding.
    
    * 4.1.alpha   New parser based on the sympy framework
    * 4.1.beta    Improved setup for spike coding, in the sense of supporting spike events. Connection patterns are now a part of Projection class.

New / updated features in current revision
==============================================

Added
----------------------------------------------

    - new connection patterns - fixed_number_pre, fixed_number_post, connection_from_list
    - learning of projections can be disabled / enabled during simulation
    - structural plasticity for spike coded neurons

Changed
----------------------------------------------

    - interface for connection patterns
    - calling of connection patterns

Removed
----------------------------------------------

    - visualization prototype based on matplotlib

Bugfix
----------------------------------------------

    - structural plasticity for mean rate coded neurons
    - storage as matlab files
    - rank, value, delay are now accessible by their dendrite

Features coming soon (version 4.2)
====================================

    - GPGPU implementation
    - weight sharing
