"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
from ANNarchy import add_function, Constant, Hebb, IF_curr_exp, \
    Network, Neuron, STDP, STP, Synapse, Uniform


add_function("fa(x) = 2/(x*x)")
mu = Constant('mu', 0.2)

Constant('b', 0.2)
add_function("f(x) = 5.0 * x + 140")


Izhikevich = Neuron(
    parameters = """
        noise = 0.0
        a = 0.02 : population
        c = -65.0
        d = 8.0
        v_thresh = 30.0
        i_offset = 0.0
    """,
    equations = """
        I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset
        dv/dt = f2(v) + f(v) - u + I : init = -65.0
        du/dt = a * (b*v - u) : init= -13.0
    """,
    spike = "v > v_thresh",
    reset = "v = c; u += d",
    refractory = 0.0,
    functions="f2(x) = 0.04 * x^2"
)

DefaultNeuron = Neuron(
    parameters = """
        tau = 10.0 : population
        baseline = 1.0 : population
        condition = true : bool
    """,
    equations= """
        noise = Uniform(-0.1, 0.1)
        tau * dmp/dt + mp = fa(sum(exc)) +  neg(- fb(sum(inh)))
                            + baseline + noise : implicit
        r = pos(mp)
    """,
    functions="""
        fb(x) = if x > 0: x else:x/2
    """
)

emptyNeuron1 = Neuron(equations="r=0")
emptyNeuron2 = Neuron(parameters="r=0")

Oja = Synapse(
    parameters = """
        eta = 10.0
        tau = 10.0 : projection
    """,
    equations = """
        tau * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
        eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
        mu * dx/dt = pre.sum(exc) * post.sum(exc)
    """,
    psp = """
        w * pre.r
    """
)

class RateCodedNetwork(Network):
    def __init__(self):

        self.pop1 = self.create(name='pop1', neuron=emptyNeuron1, geometry=1)
        self.pop2 = self.create(name='pop2', neuron=DefaultNeuron, geometry=1)
        self.pop3 = self.create(name='pop3', neuron=emptyNeuron2, geometry=(2,2))
        
        self.proj1 = self.connect(pre=self.pop1, post=self.pop2, target='exc')
        self.proj1.one_to_one(weights=Uniform(-0.5, 0.5))
        
        self.proj2 = self.connect(pre=self.pop1, post=self.pop3, target='exc', synapse=Hebb)
        self.proj2.all_to_all(1.0)
        
        self.proj3 = self.connect(pre=self.pop2, post=self.pop3, target='exc', synapse=Oja)
        self.proj3.all_to_all(1.0)

        self.m = self.monitor(self.pop2,'r')



class SpikingNetwork(Network):

    def __init__(self):

        self.pop1 = self.create(name='pop1', neuron=IF_curr_exp, geometry=1)
        self.pop2 = self.create(name='pop2', neuron=Izhikevich, geometry=1)
        self.pop3 = self.create(name='pop3', neuron=Izhikevich, geometry=(2,2))
        self.proj1 = self.connect(pre=self.pop1, post=self.pop2, target='exc', synapse=STDP)
        self.proj1.one_to_one(weights=Uniform(-0.5, 0.5))
        self.proj2 = self.connect(pre=self.pop2, post=self.pop3, target='exc', synapse=STP)
        self.proj2.all_to_all(1.0)

        m = self.monitor(self.pop2,'r')
