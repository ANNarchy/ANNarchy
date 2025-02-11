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
    Constant parameter that can be used by all neurons and synapses.

    The class `Constant` derives from `float`, so any legal operation on floats (addition, multiplication) can be used, but it returns a float.

    If a Neuron/Synapse already defines a parameter with the same name, the constant will not be visible.

    The constant can be declared at the global level, usually before the neuron/synapse definition:

    ```python
    tau = ann.Constant('tau', 20)
    factor = ann.Constant('factor', 0.1)
    real_tau = ann.Constant('real_tau', tau*factor)

    neuron = ann.Neuron(
        equations=[
            'real_tau * dr/dt + r = 1.0'
        ]
    )

    net = ann.Network()
    net.create(10, neuron)
    net.compile()
    ```

    The value of the constant can be changed anytime with the ``set()`` method. 
    
    ```python
    tau.set(30.0)
    ```

    If `tau` was defined at the global level, ALL networks using that constant will see the change. If you want the change to impact only one network, you should first retrieve the local `Constant` instance from the network:

    ```python
    tau = net.get_constant('tau')
    tau.set(30.0)
    ```
    
    Assignments will have no effect (e.g. `tau = 10.0` creates a new float and erases the `Constant` object).

    The value of constants defined as combination of other constants (`real_tau`) is not updated if the value of these constants changes (changing `tau` with `tau.set(10.0)` will not modify the value of `real_tau`).

    :param name: name of the constant (unique), which can be used in equations.
    :param value: the value of the constant, which must be a float, or a combination of Constants.
    """

    def __new__(cls, name, value, net_id=0):
        return float.__new__(cls, value)
        
    def __init__(self, name:str, value:float, net_id=0):

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
        
        If the constant was declared globally, this impacts all networks. Call `Network.get_constant(name)` if you want to impact a single network.

        :param value: Value.
        """
        self.value = value

        if NetworkManager().get_network(net_id=self.net_id).compiled:
            getattr(NetworkManager().get_network(net_id=self.net_id).instance, 'set_'+self.name)(self.value)

        for net_id in self._child_nets:
            NetworkManager().get_network(net_id=net_id).get_constant(self.name).set(value)

