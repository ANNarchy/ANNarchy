"""
ANNarchy supports different data structures to represent connection patterns between
two populations. They will differ among model type (rate-code or spiking) and more over
the target platform.

contained classes:

* OpenMPConnectivity: should extend the OpenMPGenerator
* CUDAConnectivity: should extend the CUDAGenerator

"""
from .Connectivity import OpenMPConnectivity, CUDAConnectivity
