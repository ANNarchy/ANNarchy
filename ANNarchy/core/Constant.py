"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core import Global

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern import Messages

class Constant(float):
    """
    Constant parameter that can be used by all neurons and synapses.

    The class ``Constant`` derives from ``float``, so any legal operation on floats (addition, multiplication) can be used.

    If a Neuron/Synapse defines a parameter with the same name, the constant parameters will not be visible.

    Example:

    ```python

    tau = Constant('tau', 20)
    factor = Constant('factor', 0.1)
    real_tau = Constant('real_tau', tau*factor)

    neuron = Neuron(
        equations='''
            real_tau*dr/dt + r =1.0
        '''
    )
    ```

    The value of the constant can be changed anytime with the ``set()`` method. Assignments will have no effect (e.g. ``tau = 10.0`` only creates a new float).

    The value of constants defined as combination of other constants (``real_tau``) is not updated if the value of these constants changes (changing ``tau`` with ``tau.set(10.0)`` will not modify the value of ``real_tau``).

    """
    def __new__(cls, name, value, net_id=0):
        return float.__new__(cls, value)
        
    def __init__(self, name, value, net_id=0):
        """
        :param name: name of the constant (unique), which can be used in equations.
        :param value: the value of the constant, which must be a float, or a combination of Constants.
        """

        self.name = name
        self.value = value
        self.net_id = net_id
        for obj in Global._objects['constants']:
            if obj.name == name:
                Messages._error('the constant', name, 'is already defined.')
        Global._objects['constants'].append(self)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def set(self, value):
        "Changes the value of the constant."
        self.value = value
        if NetworkManager().is_compiled(net_id=self.net_id):
            getattr(NetworkManager().cy_instance(net_id=self.net_id), '_set_'+self.name)(self.value)
