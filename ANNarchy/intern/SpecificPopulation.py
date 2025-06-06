"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Population import Population
from ANNarchy.intern.ConfigManagement import ConfigManager

class SpecificPopulation(Population):
    """
    Interface class for user-defined definition of Population objects. An inheriting
    class need to override the implementor functions _generate_[paradigm], otherwise
    a NotImplementedError exception will be thrown.
    """
    def __init__(self, geometry, neuron, name=None, copied=False, net_id=0):
        """
        Initialization, receive default arguments of Population objects.
        """
        Population.__init__(self, geometry=geometry, neuron=neuron, name=name, stop_condition=None, storage_order='post_to_pre', copied=copied, net_id=net_id)

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

    def _generate_st(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread.
        """
        raise NotImplementedError

    def _generate_omp(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for openMP paradigm.
        """
        raise NotImplementedError

    def _generate_cuda(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread and openMP paradigm.
        """
        raise NotImplementedError
