"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Projection import Projection
from ANNarchy.intern.ConfigManagement import ConfigManager

class SpecificProjection(Projection):
    """
    Interface class for user-defined definition of Projection objects. An inheriting
    class need to override the implementor functions _generate_[paradigm], otherwise
    a NotImplementedError exception will be thrown.
    """
    def __init__(self, pre, post, target, synapse=None, name=None, copied=False, net_id=0):
        """
        Initialization, receive parameters of Projection objects.

        :param pre: pre-synaptic population.
        :param post: post-synaptic population.
        :param target: type of the connection.
        :param window: duration of the time window to collect spikes (default: dt).
        """
        Projection.__init__(self, pre=pre, post=post, target=target, synapse=synapse, name=name, copied=copied, net_id=net_id)

    def _generate(self):
        """
        Overridden method of Population, called during the code generation process.
        This function selects dependent on the chosen paradigm the correct implementor
        functions defined by the user.
        """
        if ConfigManager().get('paradigm', self.net_id) == "openmp":
            if ConfigManager().get('num_threads', self.net_id) == 1:
                self._generate_st()
            else:
                self._generate_omp()
        elif ConfigManager().get('paradigm', self.net_id) == "cuda":
            self._generate_cuda()
        else:
            raise NotImplementedError

    def _generate_omp(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread and openMP paradigm.
        """
        raise NotImplementedError

    def _generate_cuda(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for CUDA paradigm.
        """
        raise NotImplementedError




