"""
The module provides an implementation of a BOLD signal monitor to record from a spiking
population.

The work is based on the work of Friston et al. more details can be found in the description
of the BoldMonitor class. To record from a specific population, we need two kinds of normalization
stages. First a normalization of received conductance (done in *NormProjection*). Those conductances
are then summed up for all neurons in the recorded population which requires also a normalization (this
stage is implemented in *AccProjection*).

Often neurons receive connections from multiple targets which requires the summation of all
afferent connections. This number of afferent synapses is accumulated and set by the function
*update_num_aff_connections*.
"""
from .BoldMonitor import BoldMonitor
from .AccProjection import AccProjection
from .NormProjection import NormProjection

__all__ = ['BoldMonitor', 'AccProjection', 'NormProjection']