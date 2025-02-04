"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager

def get_constant(name):
    """
    Returns the Constant object with the given name, None otherwise.
    """
    return  NetworkManager().get_network(net_id=0).get_constant(name)

class Constant(float):
    """
    Constant parameter that can be used by all neurons and synapses used in a network.

    See `Network.constant()`.
    """
    def __new__(cls, name, value, net_id=0):
        return float.__new__(cls, value)
        
    def __init__(self, name, value, net_id=0):

        self.name = name
        "Name."
        self.value = value
        "Value."

        self.net_id = net_id
        self._child_nets = []

        NetworkManager().get_network(net_id=net_id)._add_constant(self)

    def _copy(self, net_id=None):
        return Constant(name=self.name, value=self.value, net_id=self.net_id if net_id is None else net_id)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()
    
    def _set_child_network(self, net_id) -> None:
        "Tells the constant (of id 0) to also update the ones in networks on a change."
        self._child_nets.append(net_id)

    def set(self, value:float) -> None:
        """
        Changes the value of the constant.

        :param value: Value.
        :param network: Network instance which should be updated. By default, all active networks are updated.
        """
        self.value = value

        if NetworkManager().get_network(net_id=self.net_id).compiled:
            getattr(NetworkManager().get_network(net_id=self.net_id).instance, 'set_'+self.name)(self.value)

        for net_id in self._child_nets:
            NetworkManager().get_network(net_id=net_id).get_constant(self.name).set(value)

