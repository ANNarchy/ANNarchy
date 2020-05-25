"""
ANNarchy uses a distinct set of weights for each post-synaptic neuron (fully-connected).
However, operations like pooling or convolution are applied with the same weights for all pre-neurons within
a projection. Consequently, the required weight matrix need to be stored only once. This can
save computation time and especially memory space.

Contained classes:

* Convolution: simple-, layer-wise and bank of filter convolution
* Pooling: pooling operation (max, mean, etc)
* CopyProjection: mirrors the weights of another projection.

Implementation note:

* the implementation code was formerly in weightsharing.SharedProjection.
"""
from .Pooling import Pooling
from .Convolve import Convolution
from .Copy import Copy

# Export only the instantiable classes
__all__ = ['Pooling', 'Convolution', 'Copy']
