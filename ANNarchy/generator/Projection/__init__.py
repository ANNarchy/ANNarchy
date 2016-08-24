"""
The Projection module ...

Each Generator subclasses next to the ProjectionGenerator also a Connectivity module. The first is responsible for handling equations and
buisness logic, the latter is responsible for define data structures and provide access methods as well as indices for accessing.
"""
from .OpenMPGenerator import OpenMPGenerator as OpenMPProjectionGenerator
from .CUDAGenerator import CUDAGenerator as CUDAProjectionGenerator