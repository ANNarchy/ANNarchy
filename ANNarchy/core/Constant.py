"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.GlobalObjects import GlobalObjectManager

def get_constant(name):
    """
    Returns the Constant object with the given name, None otherwise.
    """
    return GlobalObjectManager().get_constant(name)

class Constant(float):
    """
    Constant parameter that can be used by all neurons and synapses.

    The class ``Constant`` derives from ``float``, so any legal operation on floats (addition, multiplication) can be used.

    If a Neuron/Synapse defines a parameter with the same name, the constant parameters will not be visible.

    Example:

    ```python

    tau = ann.Constant('tau', 20)
    factor = ann.Constant('factor', 0.1)
    real_tau = ann.Constant('real_tau', tau*factor)

    neuron = ann.Neuron(
        equations='''
            real_tau*dr/dt + r =1.0
        '''
    )
    ```

    The value of the constant can be changed anytime with the ``set()`` method. Assignments will have no effect (e.g. ``tau = 10.0`` only creates a new float).

    The value of constants defined as combination of other constants (``real_tau``) is not updated if the value of these constants changes (changing ``tau`` with ``tau.set(10.0)`` will not modify the value of ``real_tau``).

    :param name: name of the constant (unique), which can be used in equations.
    :param value: the value of the constant, which must be a float, or a combination of Constants.
    """
    def __new__(cls, name, value):
        return float.__new__(cls, value)
        
    def __init__(self, name, value):
        """
        Constructor, implicitly calls __new__()
        """
        self.name = name
        "Name."
        self.value = value
        "Value."

        GlobalObjectManager().add_constant(self)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def set(self, value:float, network=None) -> None:
        """
        Changes the value of the constant.

        :param value: Value.
        :param network: Network instance which should be updated. By default, all active networks are updated.
        """
        self.value = value

        # change for all active networks
        if network is None:
            for net_id in NetworkManager()._get_network_ids():
                if NetworkManager().is_compiled(net_id=net_id):
                    getattr(NetworkManager().cy_instance(net_id=net_id), '_set_'+self.name)(self.value)

        # update an individual network
        else:
            if NetworkManager().is_compiled(net_id=network.id):
                getattr(NetworkManager().cy_instance(net_id=network.id), '_set_'+self.name)(self.value)
            else:
                raise ValueError("Setting a constant for individual networks is only allowed afer compilation.")
